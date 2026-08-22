// F8-2: the W8A16 M curve on the protected FP8 production shapes.
//
// D-F8-GATE requires >=82% of the local read roofline at M in {1,2,3,4},
// >=81% at M=8 and >=64% at M=16, on every eligible protected production
// shape. Experiment 0158 established the composed five-CTA scheduler at M=1;
// this probe measures the register-fed W8A16 primitive per shape across the M
// bands, which is what the gate names.
//
// Format is the checkpoint's, unchanged: E4M3 codes, E8M0 block-128 scales,
// BF16 activations at the MMA boundary. Decode, fragment order and scale
// indexing match the concurrent session's scheduler exactly so the two are
// comparable.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kWarp = 32U;
constexpr std::uint32_t kThreads = 128U;
constexpr std::uint32_t kWarps = kThreads / kWarp;
constexpr std::uint32_t kWarmups = 3U;
constexpr std::uint32_t kSamples = 11U;
constexpr std::size_t kScrubBytes = 256ULL << 20U;
constexpr std::size_t kRulerBytes = 128ULL << 20U;
constexpr std::size_t kCeiling = 512ULL << 20U;

// split is the K decomposition each shape needs to fill the machine. Without
// it the small shapes starve: wkv has only 32 N-tiles, which is 0.10 waves per
// SM and measured 7.5% of roofline. Values match the concurrent session's
// scheduler, which reached them independently.
struct Shape { const char* name; std::uint32_t n, k, split; };
constexpr Shape kShapes[] = {
    {"wq_a", 1024U, 4096U, 16U},  {"wq_b", 40960U, 1024U, 1U},
    {"wkv", 512U, 4096U, 16U},    {"wo_a", 8192U, 4096U, 4U},
    {"wo_b", 4096U, 8192U, 8U},
};

void check(cudaError_t s, std::string_view op) {
    if (s != cudaSuccess)
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(s));
}

class Buffer {
  public:
    explicit Buffer(std::size_t b) : bytes_(b) { check(cudaMalloc(&p_, b), "alloc"); }
    ~Buffer() { if (p_) static_cast<void>(cudaFree(p_)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    [[nodiscard]] void* get() const noexcept { return p_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  private:
    void* p_{nullptr}; std::size_t bytes_{0};
};

std::uint32_t rng(std::uint32_t& s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
std::uint16_t bf16_bits(float v) {
    std::uint32_t b; std::memcpy(&b,&v,4);
    return std::uint16_t((b + 0x7FFFU + ((b>>16)&1U)) >> 16);
}
float bf16_value(std::uint16_t b) {
    std::uint32_t w = std::uint32_t(b) << 16; float f; std::memcpy(&f,&w,4); return f;
}
// E4M3 and E8M0 exactly as the scheduler's host oracle defines them.
float e4(std::uint8_t c) {
    auto e=(c>>3)&15U, m=c&7U;
    float v = e ? std::ldexp(1.0F+float(m)/8.0F,int(e)-7) : std::ldexp(float(m),-9);
    return (c&0x80U)?-v:v;
}
float e8(std::uint8_t s) { return std::ldexp(1.0F,int(s)-127); }

// Matches the scheduler's device rounder exactly, so both publish identical
// BF16 for identical FP32 input.
__device__ __forceinline__ std::uint16_t round_bf16(float x) {
    std::uint32_t b = __float_as_uint(x);
    b += 0x7FFFU + ((b>>16)&1U);
    return std::uint16_t(b>>16);
}

__device__ __forceinline__ std::uint32_t decode_pair(std::uint32_t pair,
                                                     std::uint32_t factor) {
    const auto p = __byte_perm(pair,0U,0x4140U);
    const std::uint32_t v = ((p<<8)&0x8000'8000U) | ((p<<4)&0x07F0'07F0U);
    const auto r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&v),
                           *reinterpret_cast<const __nv_bfloat162*>(&factor));
    return *reinterpret_cast<const std::uint32_t*>(&r);
}
__device__ __forceinline__ void mma(float&d0,float&d1,float&d2,float&d3,
    std::uint32_t a0,std::uint32_t a1,std::uint32_t a2,std::uint32_t a3,
    std::uint32_t b0,std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

// M activation columns. CB column blocks of 8; ACCS independent accumulator
// sets per block, tree-combined, which is the association fix that keeps the
// FP32 chain short over deep K.
template <int M, int CB, int ACCS>
__global__ __launch_bounds__(kThreads, 4) void project_kernel(
    const uint4* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const uint2* __restrict__ x, float* __restrict__ partials,
    std::uint16_t* __restrict__ y, std::uint32_t n, std::uint32_t k,
    std::uint32_t split) {
    const std::uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U;
    const std::uint32_t group = lane >> 2U, thread = lane & 3U;
    const std::uint32_t kpairs = k / 32U;
    const std::uint32_t tiles = n / 16U, work = tiles * split;
    const std::uint32_t per_slice = kpairs / split;
    for (std::uint32_t w = blockIdx.x * kWarps + warp; w < work;
         w += gridDim.x * kWarps) {
        const std::uint32_t nt = w / split, slice = w % split;
        const std::uint32_t kbegin = slice * per_slice, kend = kbegin + per_slice;
        float acc[CB][ACCS][4]{};
        for (std::uint32_t pb = kbegin; pb < kend; pb += 4U * ACCS / 2U) {
#pragma unroll
            for (int chunk = 0; chunk < ACCS / 2; ++chunk) {
                const std::uint32_t cpb = pb + std::uint32_t(chunk) * 4U;
                if (cpb >= kend) break;
                const std::uint32_t ktb = cpb * 2U;
                std::uint32_t s = lane == 0U
                    ? scales[(nt/8U)*(k/128U) + ktb/8U] : 0U;
                s = __shfl_sync(0xFFFF'FFFFU, s, 0);
                const std::uint32_t factor = ((s + 120U) << 7U) * 0x0001'0001U;
                uint4 q[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    q[u] = codes[(std::size_t(nt)*kpairs + cpb + u)*kWarp + lane];
#pragma unroll
                for (int u = 0; u < 4; ++u)
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const std::uint32_t lo = j==0?q[u].x:q[u].z,
                                            hi = j==0?q[u].y:q[u].w;
                        const auto a0=decode_pair(lo&0xFFFFU,factor),
                                   a1=decode_pair(lo>>16,factor),
                                   a2=decode_pair(hi&0xFFFFU,factor),
                                   a3=decode_pair(hi>>16,factor);
                        const std::uint32_t kt = ktb + u*2U + j;
                        const int idx = chunk*2 + j;
#pragma unroll
                        for (int c = 0; c < CB; ++c) {
                            // Activations are pre-permuted into B-fragment
                            // order once, so each lane does a single 8-byte
                            // load per column block regardless of M. Reading
                            // [k][M] directly costs M strided 16-bit load pairs
                            // per MMA, which is what made the curve fall with M.
                            const uint2 b = x[(std::size_t(kt)*CB + c)*kWarp + lane];
                            auto& a = acc[c][idx];
                            mma(a[0],a[1],a[2],a[3],a0,a1,a2,a3,b.x,b.y);
                        }
                    }
            }
        }
#pragma unroll
        for (int c = 0; c < CB; ++c)
#pragma unroll
            for (int step = 1; step < ACCS; step *= 2)
#pragma unroll
                for (int base = 0; base < ACCS; base += step*2)
#pragma unroll
                    for (int reg = 0; reg < 4; ++reg)
                        acc[c][base][reg] += acc[c][base+step][reg];
        // D: row = group + (i>=2?8:0), column = c*8 + thread*2 + (i&1).
        // Only live (row, col) pairs are stored: at M=1 a full 16x8 tile would
        // make 7/8 of the split-K partial traffic zeros.
        // With split == 1 the warp already owns all of K, so the partial
        // round-trip is pure waste: publish straight to the output instead.
        float* slot = partials + (std::size_t(nt)*split + slice)*16U*M;
#pragma unroll
        for (int c = 0; c < CB; ++c)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const std::uint32_t row = group + (i>=2?8U:0U);
                const std::uint32_t col = c*8U + thread*2U + (i&1);
                if (col >= M) continue;
                if (split == 1U)
                    y[(std::size_t(nt)*16U + row)*M + col] = round_bf16(acc[c][0][i]);
                else
                    slot[row*M + col] = acc[c][0][i];
            }
    }
}

__global__ void reduce_kernel(const float* __restrict__ partials,
                              std::uint16_t* __restrict__ y, std::uint32_t n,
                              std::uint32_t m, std::uint32_t split) {
    const std::uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= n*m) return;
    const std::uint32_t row = index / m, col = index % m;
    const std::uint32_t nt = row / 16U, local = row % 16U;
    float sum = 0.0F;
    for (std::uint32_t s = 0U; s < split; ++s)
        sum += partials[(std::size_t(nt)*split + s)*16U*m + local*m + col];
    y[index] = round_bf16(sum);
}

__global__ void ruler_kernel(const uint4* p, std::uint64_t nv, unsigned* sink) {
    const std::uint64_t st = std::uint64_t(gridDim.x)*blockDim.x;
    unsigned a = 0U;
    for (std::uint64_t i = std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i < nv; i += st) { auto v = p[i]; a ^= v.x^v.y^v.z^v.w; }
    if (a == 0xDEAD'BEEFU) sink[threadIdx.x] = a;
}
__global__ void scrub_kernel(std::uint8_t* d, std::size_t b) {
    const std::size_t st = std::size_t(gridDim.x)*blockDim.x;
    for (std::size_t i = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i < b; i += st) d[i] = std::uint8_t(i);
}

template <class F> float timed(cudaEvent_t a, cudaEvent_t b, cudaStream_t s, F f) {
    check(cudaEventRecord(a,s),"start"); f(); check(cudaEventRecord(b,s),"stop");
    check(cudaEventSynchronize(b),"sync");
    float ms; check(cudaEventElapsedTime(&ms,a,b),"elapsed"); return ms*1000.0F;
}
float median(std::vector<float> v) {
    std::sort(v.begin(), v.end()); return v[v.size()/2];
}

// Canonical E4M3 weights plus block-128 E8M0 scales, prepacked into the same
// fragment order the scheduler uses.
struct HostMatrix {
    std::uint32_t n, k;
    std::vector<std::uint8_t> canon, scales;
    std::vector<uint4> packed;
    HostMatrix(std::uint32_t nn, std::uint32_t kk, std::uint32_t& state)
        : n(nn), k(kk), canon(std::size_t(n)*k),
          scales(std::size_t(n/128U)*(k/128U)),
          packed(std::size_t(n/16U)*(k/32U)*32U) {
        for (auto& c : canon) { c = std::uint8_t(rng(state));
                                if ((c & 0x7FU) == 0x7FU) c ^= 1U; }
        for (auto& s : scales) s = std::uint8_t(114U + rng(state) % 7U);
        for (std::uint32_t nt = 0; nt < n/16U; ++nt)
            for (std::uint32_t kp = 0; kp < k/32U; ++kp)
                for (std::uint32_t lane = 0; lane < 32U; ++lane) {
                    std::uint32_t w[4]{};
                    for (std::uint32_t j = 0; j < 2U; ++j)
                        for (std::uint32_t i = 0; i < 8U; ++i) {
                            const bool high = i==2||i==3||i==6||i==7;
                            const auto row = nt*16U + (lane>>2) + (high?8U:0U);
                            const auto col = (kp*2U+j)*16U + (lane&3U)*2U
                                             + (i&1U) + (i>=4U?8U:0U);
                            w[j*2U + (i>=4U)] |=
                                std::uint32_t(canon[std::size_t(row)*k+col])
                                << ((i&3U)*8U);
                        }
                    packed[(std::size_t(nt)*(k/32U)+kp)*32+lane] =
                        make_uint4(w[0],w[1],w[2],w[3]);
                }
    }
    [[nodiscard]] std::size_t useful_bytes() const {
        return canon.size() + scales.size();
    }
};

struct Error { std::uint32_t bad{}; double norm{}; };

Error oracle(const HostMatrix& w, const std::vector<std::uint16_t>& x,
             const std::vector<std::uint16_t>& y, std::uint32_t m) {
    Error r;
    for (std::uint32_t n = 0; n < w.n; ++n)
        for (std::uint32_t c = 0; c < m; ++c) {
            double sum = 0.0, ab = 0.0;
            for (std::uint32_t k = 0; k < w.k; ++k) {
                const auto s = w.scales[(n/128U)*(w.k/128U) + k/128U];
                const double p =
                    double(bf16_value(bf16_bits(e4(w.canon[std::size_t(n)*w.k+k])*e8(s))))
                    * bf16_value(x[std::size_t(k)*m + c]);
                sum += p; ab += std::abs(p);
            }
            const auto expect = bf16_bits(float(sum));
            const auto got = y[std::size_t(n)*m + c];
            if (got != expect) ++r.bad;
            r.norm = std::max(r.norm,
                std::abs(double(bf16_value(got)) - bf16_value(expect))
                / std::max(ab, 1e-30));
        }
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string out; std::uint32_t m = 1U; int device = 0;
        // Split-K partial traffic scales as split x M while buying identical
        // parallelism, so the optimum split falls as M rises.
        std::uint32_t split_divisor = 1U;
        for (int i = 1; i < argc; ++i) {
            std::string_view f = argv[i];
            if (f == "--output" && ++i < argc) out = argv[i];
            else if (f == "--m" && ++i < argc) m = std::uint32_t(std::stoul(argv[i]));
            else if (f == "--device" && ++i < argc) device = std::stoi(argv[i]);
            else if (f == "--split-divisor" && ++i < argc)
                split_divisor = std::uint32_t(std::stoul(argv[i]));
            else throw std::runtime_error("usage: --output F [--m M] [--device D]");
        }
        if (m!=1&&m!=2&&m!=3&&m!=4&&m!=8&&m!=16)
            throw std::runtime_error("M must be one of 1,2,3,4,8,16");
        check(cudaSetDevice(device),"device");
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,device),"props");
        if (prop.major!=8||prop.minor!=6) throw std::runtime_error("requires SM86");

        cudaStream_t st; check(cudaStreamCreate(&st),"stream");
        Buffer scrub(kScrubBytes), ruler(kRulerBytes), sink(1024);
        check(cudaMemsetAsync(ruler.get(),0xA5,ruler.bytes(),st),"ruler init");

        std::ostringstream rows; std::uint64_t useful_total = 0;
        double worst_eff = 1e9; std::uint32_t total_bad = 0; double worst_norm = 0;
        std::size_t peak = scrub.bytes()+ruler.bytes()+sink.bytes();
        float ruler_us = 0.0F;
        cudaEvent_t ea, eb; check(cudaEventCreate(&ea),"ev"); check(cudaEventCreate(&eb),"ev");
        const auto doscrub = [&]{ scrub_kernel<<<65536,256,0,st>>>((std::uint8_t*)scrub.get(),scrub.bytes()); };
        const auto doruler = [&]{ ruler_kernel<<<prop.multiProcessorCount*8,256,0,st>>>((const uint4*)ruler.get(),ruler.bytes()/16,(unsigned*)sink.get()); };

        // All five shapes stay resident and are launched as one sequence,
        // then that sequence is chained. The projections are individually far
        // too small to amortise dispatch -- wkv is 2.1 MB, about 2.5 us of work
        // against a ~4 us launch floor -- which is why the production path runs
        // them inside one persistent kernel. Timing them one launch at a time
        // measures the launch floor, not the primitive.
        constexpr std::uint32_t kChain = 8U;
        std::vector<std::unique_ptr<HostMatrix>> hosts;
        std::vector<std::vector<std::uint16_t>> xs;
        std::vector<std::vector<std::uint32_t>> xfs;
        std::vector<std::unique_ptr<Buffer>> dcs, dss, dxs, dys, dps;
        std::vector<std::uint32_t> splits;
        for (const Shape& sh : kShapes)
            splits.push_back(std::max(1U, sh.split / split_divisor));
        std::size_t shape_index = 0;
        for (const Shape& sh : kShapes) {
            const std::uint32_t split = splits[shape_index++];
            std::uint32_t state = 0x514A'9C31U ^ sh.n ^ (sh.k<<7);
            hosts.push_back(std::make_unique<HostMatrix>(sh.n, sh.k, state));
            std::vector<std::uint16_t> x(std::size_t(sh.k)*m);
            for (auto& v : x) v = bf16_bits(float(int(rng(state)%17)-8)/8.0F);
            xs.push_back(std::move(x));
            // Pre-permute into B-fragment order: lane (group, thread) of column
            // block c holds rows thread*2{,+1} and {+8,+9} of column c*8+group.
            const std::uint32_t cb = (m + 7U) / 8U, ktiles = sh.k / 16U;
            std::vector<std::uint32_t> xf(std::size_t(ktiles)*cb*kWarp*2U, 0U);
            for (std::uint32_t kt = 0; kt < ktiles; ++kt)
                for (std::uint32_t c = 0; c < cb; ++c)
                    for (std::uint32_t lane = 0; lane < kWarp; ++lane) {
                        const std::uint32_t g = lane>>2, t = lane&3U;
                        const std::uint32_t col = c*8U + g;
                        if (col >= m) continue;
                        const auto at = [&](std::uint32_t r) {
                            return std::uint32_t(
                                xs.back()[(std::size_t(kt)*16U + r)*m + col]);
                        };
                        const std::size_t o = ((std::size_t(kt)*cb + c)*kWarp + lane)*2U;
                        xf[o]   = at(t*2U)      | (at(t*2U+1U) << 16U);
                        xf[o+1] = at(t*2U+8U)   | (at(t*2U+9U) << 16U);
                    }
            xfs.push_back(std::move(xf));
            const auto& w = *hosts.back();
            dcs.push_back(std::make_unique<Buffer>(w.packed.size()*sizeof(uint4)));
            dss.push_back(std::make_unique<Buffer>(w.scales.size()));
            dxs.push_back(std::make_unique<Buffer>(xfs.back().size()*4));
            dys.push_back(std::make_unique<Buffer>(std::size_t(sh.n)*m*2));
            dps.push_back(std::make_unique<Buffer>(
                std::size_t(sh.n/16U)*split*16U*m*sizeof(float)));
            peak += dcs.back()->bytes()+dss.back()->bytes()+dxs.back()->bytes()
                    +dys.back()->bytes()+dps.back()->bytes();
            if (peak > kCeiling) throw std::runtime_error("exceeds 512 MiB");
            check(cudaMemcpyAsync(dcs.back()->get(),w.packed.data(),dcs.back()->bytes(),cudaMemcpyHostToDevice,st),"codes");
            check(cudaMemcpyAsync(dss.back()->get(),w.scales.data(),dss.back()->bytes(),cudaMemcpyHostToDevice,st),"scales");
            check(cudaMemcpyAsync(dxs.back()->get(),xfs.back().data(),dxs.back()->bytes(),cudaMemcpyHostToDevice,st),"x");
            useful_total += w.useful_bytes();
        }
        check(cudaStreamSynchronize(st),"upload");

        const std::uint32_t blocks = prop.multiProcessorCount*4;
        const auto launch_shape = [&](std::size_t i) {
            const Shape& sh = kShapes[i];
            const auto* c=(const uint4*)dcs[i]->get();
            const auto* s=(const std::uint8_t*)dss[i]->get();
            const auto* xi=(const uint2*)dxs[i]->get();
            auto* yo=(std::uint16_t*)dys[i]->get(); auto* pp=(float*)dps[i]->get();
            switch (m) {
              case 1: project_kernel<1,1,8><<<blocks,kThreads,0,st>>>(c,s,xi,pp,yo,sh.n,sh.k,splits[i]); break;
              case 2: project_kernel<2,1,8><<<blocks,kThreads,0,st>>>(c,s,xi,pp,yo,sh.n,sh.k,splits[i]); break;
              case 3: project_kernel<3,1,8><<<blocks,kThreads,0,st>>>(c,s,xi,pp,yo,sh.n,sh.k,splits[i]); break;
              case 4: project_kernel<4,1,8><<<blocks,kThreads,0,st>>>(c,s,xi,pp,yo,sh.n,sh.k,splits[i]); break;
              case 8: project_kernel<8,1,8><<<blocks,kThreads,0,st>>>(c,s,xi,pp,yo,sh.n,sh.k,splits[i]); break;
              default: project_kernel<16,2,8><<<blocks,kThreads,0,st>>>(c,s,xi,pp,yo,sh.n,sh.k,splits[i]); break;
            }
            if (splits[i] > 1U)
                reduce_kernel<<<(sh.n*m+255U)/256U,256U,0,st>>>(pp,yo,sh.n,m,splits[i]);
        };
        const auto run_all = [&]{ for (std::size_t i=0;i<std::size(kShapes);++i) launch_shape(i); };

        for (std::uint32_t i=0;i<kWarmups;++i){ doruler(); run_all(); }
        check(cudaStreamSynchronize(st),"warmup");
        std::vector<float> rt, kt;
        for (std::uint32_t i=0;i<kSamples;++i) {
            doscrub(); rt.push_back(timed(ea,eb,st,doruler));
            doscrub();
            kt.push_back(timed(ea,eb,st,[&]{ for(std::uint32_t r=0;r<kChain;++r) run_all(); })
                         / float(kChain));
        }
        check(cudaGetLastError(),"kernel");
        ruler_us = median(rt);
        const float composed_us = median(kt);

        for (std::size_t i=0;i<std::size(kShapes);++i) {
            const Shape& sh = kShapes[i]; const auto& w = *hosts[i];
            std::vector<std::uint16_t> y(std::size_t(sh.n)*m);
            check(cudaMemcpyAsync(y.data(),dys[i]->get(),y.size()*2,cudaMemcpyDeviceToHost,st),"y");
            check(cudaStreamSynchronize(st),"download");
            const auto err = oracle(w,xs[i],y,m);
            total_bad += err.bad; worst_norm = std::max(worst_norm, err.norm);
            rows << "    {\"shape\": \"" << sh.name << "\", \"n\": " << sh.n
                 << ", \"k\": " << sh.k << ", \"split\": " << splits[i]
                 << ", \"useful_bytes\": " << w.useful_bytes()
                 << ", \"mismatches\": " << err.bad
                 << ", \"max_error_over_sum_abs\": " << std::setprecision(9) << err.norm
                 << "}" << (i+1==std::size(kShapes) ? "\n" : ",\n");
        }
        const double rgb = double(kRulerBytes)/(ruler_us*1e-6)/1e9;
        const double gb = double(useful_total)/(composed_us*1e-6)/1e9;
        worst_eff = gb/rgb;
        std::ostringstream j;
        j << std::fixed << "{\n  \"device_name\": \"" << prop.name
          << "\", \"device_index\": " << device << ",\n  \"m\": " << m
          << ", \"useful_bytes_total\": " << useful_total
          << ", \"peak_device_bytes\": " << peak << ",\n  \"ruler_us\": "
          << std::setprecision(3) << ruler_us << ", \"ruler_gbps\": "
          << std::setprecision(2) << rgb << ",\n  \"composed_us\": "
          << std::setprecision(3) << composed_us
          << ", \"composed_gbps\": " << std::setprecision(2) << gb
          << ",\n  \"roofline_efficiency\": "
          << std::setprecision(6) << worst_eff << ", \"total_mismatches\": "
          << total_bad << ", \"worst_error_over_sum_abs\": "
          << std::setprecision(9) << worst_norm << ",\n  \"shapes\": [\n"
          << rows.str() << "  ]\n}\n";
        if (out.empty()) std::cout << j.str();
        else { std::ofstream f(out); if(!f) throw std::runtime_error("open "+out); f << j.str(); }
        cudaEventDestroy(ea); cudaEventDestroy(eb); cudaStreamDestroy(st);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'; return EXIT_FAILURE;
    }
}
