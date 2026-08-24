// Production-shaped SM86 QPN8-derived W8A16 kernel probe.
// E4M3/E8M0 weights stay byte-resident, are pre-permuted without arithmetic,
// decoded directly into BF16 HMMA registers, and reduced in one launch.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr std::uint32_t kWarp = 32, kTileN = 16, kTileK = 16;
constexpr std::uint32_t kWarpsPerBlock = 4, kWarmups = 3, kSamples = 11;
constexpr std::size_t kMiB = 1ULL << 20, kScrubBytes = 256ULL * kMiB;
constexpr std::size_t kRulerBytes = 128ULL * kMiB, kArenaBudget = 88ULL * kMiB;
constexpr std::size_t kCeiling = 512ULL * kMiB;

void check(cudaError_t s, std::string_view what) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s));
}
class Buffer {
  public:
    explicit Buffer(std::size_t n) : n_(n) { check(cudaMalloc(&p_, n), "cudaMalloc QPN8"); }
    ~Buffer() { if (p_) static_cast<void>(cudaFree(p_)); }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    void* get() const { return p_; } std::size_t bytes() const { return n_; }
  private: void* p_{}; std::size_t n_{};
};

std::uint32_t rng(std::uint32_t& x) { x^=x<<13; x^=x>>17; x^=x<<5; return x; }
std::uint16_t bf16_bits(float x) {
    auto b=std::bit_cast<std::uint32_t>(x); b += 0x7fffU + ((b>>16)&1U);
    return static_cast<std::uint16_t>(b>>16);
}
float bf16_value(std::uint16_t b) { return std::bit_cast<float>(std::uint32_t(b)<<16); }
float e4(std::uint8_t c) {
    const auto e=(c>>3)&15U, m=c&7U; float v;
    if (!e) v=std::ldexp(float(m),-9);
    else v=std::ldexp(1.0F+float(m)/8.0F,int(e)-7);
    return (c&0x80U)?-v:v;
}
float e8(std::uint8_t s) { return std::ldexp(1.0F,int(s)-127); }

__device__ __forceinline__ std::uint32_t decode_pair_bits(std::uint32_t pair) {
    const std::uint32_t p=__byte_perm(pair,0U,0x4140U);
    return ((p<<8)&0x80008000U)|((p<<4)&0x07f007f0U);
}
__device__ __forceinline__ std::uint32_t scale_pair(std::uint32_t v,
                                                     std::uint32_t factor) {
    const __nv_bfloat162 r=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&v),
                                   *reinterpret_cast<const __nv_bfloat162*>(&factor));
    return *reinterpret_cast<const std::uint32_t*>(&r);
}
__device__ __forceinline__ void copy_async_16(void* dst,
                                              const void* src) {
    const auto shared=static_cast<std::uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::
                 "r"(shared),"l"(src));
}
__device__ __forceinline__ void copy_async_commit() {
    asm volatile("cp.async.commit_group;\n"::);
}
__device__ __forceinline__ void mma(float& d0,float& d1,float& d2,float& d3,
                                    std::uint32_t a0,std::uint32_t a1,
                                    std::uint32_t a2,std::uint32_t a3,
                                    std::uint32_t b0,std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ float round_bf16(float x) {
    std::uint32_t b=__float_as_uint(x); b += 0x7fffU+((b>>16)&1U);
    return __uint_as_float(b&0xffff0000U);
}

template<int SPLIT, int NTILES, bool SCALE_B, bool ASYNC_COPY,
         bool SCALE_ACCUMULATOR>
__global__ void qpn8_kernel(const uint4* __restrict__ codes,
                            const std::uint8_t* __restrict__ scales,
                            const std::uint16_t* __restrict__ x,
                            float* __restrict__ partials,
                            std::uint32_t* __restrict__ counters,
                            float* __restrict__ y,
                            std::uint32_t n, std::uint32_t k,
                            std::uint32_t m, std::uint32_t batch) {
    static_assert(NTILES==1 || NTILES==2);
    static_assert(!(SCALE_B && SCALE_ACCUMULATOR));
    const std::uint32_t lane=threadIdx.x&31U, warp=threadIdx.x>>5U;
    const std::uint32_t ntiles=n/kTileN;
    const std::uint32_t ngroups=ntiles/NTILES;
    const std::uint32_t global_warp=blockIdx.x*kWarpsPerBlock+warp;
    if(global_warp>=batch*ngroups*SPLIT)return;
    const std::uint32_t work_group=global_warp/SPLIT;
    const std::uint32_t matrix=work_group/ngroups;
    const std::uint32_t ng=work_group%ngroups;
    const std::uint32_t nt=ng*NTILES;
    const std::uint32_t slice=global_warp%SPLIT;
    const std::uint32_t ktiles=k/kTileK, kpairs=ktiles/2U;
    codes+=static_cast<std::size_t>(matrix)*n*k/sizeof(uint4);
    scales+=static_cast<std::size_t>(matrix)*(n/128U)*(k/128U);
    const std::uint32_t pairs_per_slice=kpairs/SPLIT;
    const std::uint32_t pbegin=slice*pairs_per_slice, pend=pbegin+pairs_per_slice;
    const std::uint32_t group=lane>>2U, thread=lane&3U;
    float even[NTILES][4]{}, odd[NTILES][4]{};
    std::uint32_t factor=0;
    extern __shared__ __align__(16) std::uint8_t async_storage[];
    uint4* async_stage=reinterpret_cast<uint4*>(async_storage)+
        warp*(2U*NTILES*4U*kWarp);
    if constexpr(ASYNC_COPY){
#pragma unroll
        for(int tile=0;tile<NTILES;++tile)
#pragma unroll
            for(int u=0;u<4;++u){
                const uint4* src=codes+(static_cast<std::size_t>(nt+
                    std::uint32_t(tile))*kpairs+pbegin+
                    std::uint32_t(u))*kWarp+lane;
                uint4* dst=async_stage+(std::uint32_t(tile)*4U+
                    std::uint32_t(u))*kWarp+lane;
                copy_async_16(dst,src);
            }
        copy_async_commit();
    }
    // One iteration is exactly one checkpoint K128 scale block. Issue all
    // four 16-byte lane loads before consuming any, so global latency is
    // covered by four independent requests instead of the decoder waiting on
    // one request at a time.
    for (std::uint32_t pair_block=pbegin; pair_block<pend; pair_block+=4U) {
        float block_even[NTILES][4]{},block_odd[NTILES][4]{};
        const std::uint32_t kt_block=pair_block*2U;
        const std::uint32_t block_index=(pair_block-pbegin)/4U;
        const std::uint32_t stage=block_index&1U;
        if constexpr(ASYNC_COPY){
            if(pair_block+4U<pend){
#pragma unroll
                for(int tile=0;tile<NTILES;++tile)
#pragma unroll
                    for(int u=0;u<4;++u){
                        const uint4* src=codes+(static_cast<std::size_t>(nt+
                            std::uint32_t(tile))*kpairs+pair_block+4U+
                            std::uint32_t(u))*kWarp+lane;
                        uint4* dst=async_stage+(((stage^1U)*NTILES+
                            std::uint32_t(tile))*4U+std::uint32_t(u))*kWarp+
                            lane;
                        copy_async_16(dst,src);
                    }
                copy_async_commit();
            }
        }
        std::uint32_t s=lane==0U
            ? scales[(nt/8U)*(k/128U)+kt_block/8U] : 0U;
        s=__shfl_sync(0xffffffffU,s,0);
        factor=((s+120U)<<7U)*0x00010001U;
        uint4 packed[NTILES][4];
        if constexpr(ASYNC_COPY){
            if(pair_block+4U<pend)
                asm volatile("cp.async.wait_group 1;\n"::);
            else
                asm volatile("cp.async.wait_group 0;\n"::);
            __syncwarp();
        }
#pragma unroll
        for (int tile=0;tile<NTILES;++tile)
#pragma unroll
            for (int u=0;u<4;++u)
                if constexpr(ASYNC_COPY)
                    packed[tile][u]=async_stage[((stage*NTILES+
                        std::uint32_t(tile))*4U+std::uint32_t(u))*kWarp+lane];
                else
                    packed[tile][u]=codes[(static_cast<std::size_t>(nt+
                        std::uint32_t(tile))*kpairs+pair_block+
                        std::uint32_t(u))*kWarp+lane];
#pragma unroll
        for (int u=0;u<4;++u) {
#pragma unroll
        for (int j=0;j<2;++j) {
            std::uint32_t b0=0,b1=0;
            if (group<m) {
                const std::uint32_t kt=kt_block+std::uint32_t(u)*2U+
                                       std::uint32_t(j);
                const std::uint16_t* base=x+static_cast<std::size_t>(group)*k+
                    kt*kTileK+thread*2U;
                b0=*reinterpret_cast<const std::uint32_t*>(base);
                b1=*reinterpret_cast<const std::uint32_t*>(base+8U);
                if constexpr(SCALE_B){
                    b0=scale_pair(b0,factor);
                    b1=scale_pair(b1,factor);
                }
            }
#pragma unroll
            for(int tile=0;tile<NTILES;++tile){
                const uint4 q=packed[tile][u];
                const std::uint32_t lo=j==0?q.x:q.z, hi=j==0?q.y:q.w;
                std::uint32_t a0=decode_pair_bits(lo&0xffffU);
                std::uint32_t a1=decode_pair_bits(lo>>16U);
                std::uint32_t a2=decode_pair_bits(hi&0xffffU);
                std::uint32_t a3=decode_pair_bits(hi>>16U);
                if constexpr(!SCALE_B && !SCALE_ACCUMULATOR){
                    a0=scale_pair(a0,factor);a1=scale_pair(a1,factor);
                    a2=scale_pair(a2,factor);a3=scale_pair(a3,factor);
                }
                float* acc;
                if constexpr(SCALE_ACCUMULATOR)
                    acc=j==0?block_even[tile]:block_odd[tile];
                else
                    acc=j==0?even[tile]:odd[tile];
                mma(acc[0],acc[1],acc[2],acc[3],a0,a1,a2,a3,b0,b1);
            }
        }
        }
        if constexpr(SCALE_ACCUMULATOR){
            const float block_scale=__uint_as_float((s+120U)<<23U);
#pragma unroll
            for(int tile=0;tile<NTILES;++tile)
#pragma unroll
                for(int r=0;r<4;++r)
                    even[tile][r]+=(block_even[tile][r]+
                                    block_odd[tile][r])*block_scale;
        }
    }
    for(int tile=0;tile<NTILES;++tile){
#pragma unroll
        for(int r=0;r<4;++r)
            if constexpr(!SCALE_ACCUMULATOR)even[tile][r]+=odd[tile][r];
        float* dst=partials+((static_cast<std::size_t>(matrix*ntiles+nt+
            std::uint32_t(tile))*SPLIT+slice)*128U+lane*4U);
        dst[0]=even[tile][0];dst[1]=even[tile][1];
        dst[2]=even[tile][2];dst[3]=even[tile][3];
    }
    __threadfence();
    __shared__ std::uint32_t arrived[kWarpsPerBlock];
    if(lane==0U)arrived[warp]=atomicAdd(counters+matrix*ngroups+ng,1U);
    __syncwarp();
    if(arrived[warp]!=SPLIT-1U)return;
    for(int tile=0;tile<NTILES;++tile)
        for(std::uint32_t elem=lane;elem<kTileN*m;elem+=kWarp){
            float sum=0;
#pragma unroll
            for(std::uint32_t sk=0;sk<SPLIT;++sk){
                const std::uint32_t nr=elem/m,mc=elem%m;
                const std::uint32_t g=nr&7U,upper=nr>>3U,t=mc>>1U,
                                    odd_reg=mc&1U;
                const std::uint32_t source_lane=g*4U+t,
                                    reg=upper*2U+odd_reg;
                sum+=partials[(static_cast<std::size_t>(matrix*ntiles+nt+
                    std::uint32_t(tile))*SPLIT+sk)*128U+
                    source_lane*4U+reg];
            }
            const std::uint32_t nr=elem/m,mc=elem%m;
            y[static_cast<std::size_t>(matrix)*m*n+
                static_cast<std::size_t>(mc)*n+(nt+
                std::uint32_t(tile))*kTileN+nr]=round_bf16(sum);
        }
    if(lane==0U)counters[matrix*ngroups+ng]=0U;
}

__global__ void scrub_kernel(std::uint8_t* p,std::size_t n) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x,
        s=std::size_t(gridDim.x)*blockDim.x;i<n;i+=s)p[i]=std::uint8_t(i);
}
__global__ void ruler_kernel(const uint4* p,std::uint64_t n,unsigned* sink) {
    const std::uint64_t s=std::uint64_t(gridDim.x)*blockDim.x;
    std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    unsigned a=0,b=0,c=0,d=0;
    for(;i+3*s<n;i+=4*s){auto x=p[i],y=p[i+s],z=p[i+2*s],w=p[i+3*s];a^=x.x^x.y^x.z^x.w;b^=y.x^y.y^y.z^y.w;c^=z.x^z.y^z.z^z.w;d^=w.x^w.y^w.z^w.w;}
    for(;i<n;i+=s){auto x=p[i];a^=x.x^x.y^x.z^x.w;} if((a^b^c^d)==0xdeadbeefU)sink[threadIdx.x]=a;
}
float median(std::vector<float> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
template<class F> float timed(cudaEvent_t a,cudaEvent_t b,cudaStream_t s,F f){check(cudaEventRecord(a,s),"event start");f();check(cudaEventRecord(b,s),"event stop");check(cudaEventSynchronize(b),"event sync");float ms;check(cudaEventElapsedTime(&ms,a,b),"event time");return ms*1000;}

struct Options{std::uint32_t n=512,k=4096,m=1,split=16,ntiles=1,batch=1;bool scale_b=false,async_copy=false,scale_accumulator=false;int device=0;std::string output;};
Options parse(int argc,char**argv){Options o;for(int i=1;i<argc;++i){std::string_view f=argv[i];auto val=[&](){if(++i>=argc)throw std::runtime_error("missing value");return argv[i];};if(f=="--n")o.n=std::stoul(val());else if(f=="--k")o.k=std::stoul(val());else if(f=="--m")o.m=std::stoul(val());else if(f=="--split")o.split=std::stoul(val());else if(f=="--ntiles")o.ntiles=std::stoul(val());else if(f=="--batch")o.batch=std::stoul(val());else if(f=="--scale-on-b")o.scale_b=true;else if(f=="--async-copy")o.async_copy=true;else if(f=="--scale-accumulator")o.scale_accumulator=true;else if(f=="--device")o.device=std::stoi(val());else if(f=="--output")o.output=val();else throw std::runtime_error("unknown option");}return o;}

template<int S,int NT,bool SB,bool AC,bool SA> void launch(const uint4*c,const std::uint8_t*sc,const std::uint16_t*x,float*p,std::uint32_t*ct,float*y,const Options&o,cudaStream_t st){const auto warps=o.batch*(o.n/(16U*NT))*S;const auto blocks=(warps+kWarpsPerBlock-1U)/kWarpsPerBlock;const std::size_t shared=AC?kWarpsPerBlock*2U*NT*4U*kWarp*sizeof(uint4):0U;qpn8_kernel<S,NT,SB,AC,SA><<<blocks,kWarpsPerBlock*kWarp,shared,st>>>(c,sc,x,p,ct,y,o.n,o.k,o.m,o.batch);}
template<int S,int NT> void launch_modes(const uint4*c,const std::uint8_t*sc,const std::uint16_t*x,float*p,std::uint32_t*ct,float*y,const Options&o,cudaStream_t st){if(o.scale_accumulator)launch<S,NT,false,false,true>(c,sc,x,p,ct,y,o,st);else if(o.async_copy){if(o.scale_b)launch<S,NT,true,true,false>(c,sc,x,p,ct,y,o,st);else launch<S,NT,false,true,false>(c,sc,x,p,ct,y,o,st);}else{if(o.scale_b)launch<S,NT,true,false,false>(c,sc,x,p,ct,y,o,st);else launch<S,NT,false,false,false>(c,sc,x,p,ct,y,o,st);}}
template<class F> void select_split(std::uint32_t s,F f){switch(s){case 1:f(std::integral_constant<int,1>{});break;case 2:f(std::integral_constant<int,2>{});break;case 4:f(std::integral_constant<int,4>{});break;case 8:f(std::integral_constant<int,8>{});break;case 16:f(std::integral_constant<int,16>{});break;case 32:f(std::integral_constant<int,32>{});break;default:throw std::runtime_error("split must be 1/2/4/8/16/32");}}
}

int main(int argc,char**argv){
 try{
  const auto o=parse(argc,argv); if(o.m<1||o.m>8||o.n%128||o.k%128||(o.ntiles!=1&&o.ntiles!=2)||o.batch<1||(o.scale_accumulator&&(o.scale_b||o.async_copy)))throw std::runtime_error("requires M 1..8, NTILES 1/2, batch >=1, N/K multiples of 128, and one scale/copy experiment at a time");
  check(cudaSetDevice(o.device),"set device");cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,o.device),"properties");if(prop.major!=8||prop.minor!=6)throw std::runtime_error("requires SM86");
  const std::size_t code_bytes=std::size_t(o.n)*o.k,scale_bytes=std::size_t(o.n/128)*(o.k/128),matrix_bytes=code_bytes+scale_bytes;
  std::vector<std::uint8_t> canon(code_bytes),scales(scale_bytes);std::uint32_t state=0x8f01cafeU;for(auto&c:canon){c=std::uint8_t(rng(state));if((c&0x7fU)==0x7fU)c^=1U;}for(auto&s:scales)s=std::uint8_t(114U+rng(state)%7U);
  const std::uint32_t ktiles=o.k/16U,kpairs=ktiles/2U,ntiles=o.n/16U;std::vector<uint4> packed(std::size_t(ntiles)*kpairs*kWarp);
  for(std::uint32_t nt=0;nt<ntiles;++nt)for(std::uint32_t kp=0;kp<kpairs;++kp)for(std::uint32_t lane=0;lane<32;++lane){std::uint32_t words[4]{};for(std::uint32_t j=0;j<2;++j)for(std::uint32_t i=0;i<8;++i){const bool high=i==2||i==3||i==6||i==7;const auto row=nt*16+(lane>>2)+(high?8:0);const auto col=(kp*2+j)*16+(lane&3)*2+(i&1)+(i>=4?8:0);words[j*2+(i>=4) ]|=std::uint32_t(canon[std::size_t(row)*o.k+col])<<((i&3)*8);}packed[(std::size_t(nt)*kpairs+kp)*32+lane]=make_uint4(words[0],words[1],words[2],words[3]);}
  std::vector<std::uint16_t> x(std::size_t(o.m)*o.k);for(auto&v:x)v=bf16_bits(float(int(rng(state)%17)-8)/8.0F);
  if(std::size_t(o.batch)*code_bytes>kArenaBudget)throw std::runtime_error("batch exceeds rotating arena budget");const auto replicas=std::max<std::uint32_t>(1,static_cast<std::uint32_t>(kArenaBudget/(std::size_t(o.batch)*code_bytes)));const std::size_t arena_bytes=std::size_t(replicas)*o.batch*code_bytes;std::vector<std::uint8_t> arena(arena_bytes);for(std::uint32_t r=0;r<replicas*o.batch;++r)std::memcpy(arena.data()+std::size_t(r)*code_bytes,packed.data(),code_bytes);std::vector<std::uint8_t> scale_batch(std::size_t(o.batch)*scale_bytes);for(std::uint32_t b=0;b<o.batch;++b)std::memcpy(scale_batch.data()+std::size_t(b)*scale_bytes,scales.data(),scale_bytes);
  const std::size_t partial_bytes=std::size_t(o.batch)*ntiles*o.split*128*sizeof(float),output_bytes=std::size_t(o.batch)*o.m*o.n*sizeof(float);
  Buffer da(arena_bytes),ds(scale_batch.size()),dx(x.size()*2),dp(partial_bytes),dc(std::size_t(o.batch)*(ntiles/o.ntiles)*4),dy(output_bytes),scrub(kScrubBytes),ruler(kRulerBytes),sink(1024);
  const std::size_t peak=da.bytes()+ds.bytes()+dx.bytes()+dp.bytes()+dc.bytes()+dy.bytes()+scrub.bytes()+ruler.bytes()+sink.bytes();if(peak>kCeiling)throw std::runtime_error("exceeds 512 MiB");
  cudaStream_t st;check(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");check(cudaMemcpyAsync(da.get(),arena.data(),arena_bytes,cudaMemcpyHostToDevice,st),"codes");check(cudaMemcpyAsync(ds.get(),scale_batch.data(),scale_batch.size(),cudaMemcpyHostToDevice,st),"scales");check(cudaMemcpyAsync(dx.get(),x.data(),dx.bytes(),cudaMemcpyHostToDevice,st),"x");check(cudaMemsetAsync(dc.get(),0,dc.bytes(),st),"counters");check(cudaMemsetAsync(ruler.get(),0xa5,ruler.bytes(),st),"ruler init");check(cudaStreamSynchronize(st),"uploads");
  cudaEvent_t a,b;check(cudaEventCreate(&a),"event");check(cudaEventCreate(&b),"event");auto doscrub=[&]{scrub_kernel<<<65536,256,0,st>>>((std::uint8_t*)scrub.get(),scrub.bytes());};auto doruler=[&]{ruler_kernel<<<prop.multiProcessorCount*8,256,0,st>>>((const uint4*)ruler.get(),ruler.bytes()/16,(unsigned*)sink.get());};
  auto run=[&](std::uint32_t rep){select_split(o.split,[&](auto s){const auto* c=(const uint4*)((const std::uint8_t*)da.get()+std::size_t(rep)*o.batch*code_bytes);if(o.ntiles==1)launch_modes<s.value,1>(c,(const std::uint8_t*)ds.get(),(const std::uint16_t*)dx.get(),(float*)dp.get(),(std::uint32_t*)dc.get(),(float*)dy.get(),o,st);else launch_modes<s.value,2>(c,(const std::uint8_t*)ds.get(),(const std::uint16_t*)dx.get(),(float*)dp.get(),(std::uint32_t*)dc.get(),(float*)dy.get(),o,st);});};
  for(unsigned i=0;i<kWarmups;++i){doruler();run(0);}check(cudaStreamSynchronize(st),"warmup");std::vector<float> rt,kt;for(unsigned i=0;i<kSamples;++i){doscrub();rt.push_back(timed(a,b,st,doruler));doscrub();kt.push_back(timed(a,b,st,[&]{run(i%replicas);}));}const float ru=median(rt),ku=median(kt);check(cudaGetLastError(),"kernel");
  std::vector<float> out(std::size_t(o.batch)*o.m*o.n);check(cudaMemcpy(out.data(),dy.get(),output_bytes,cudaMemcpyDeviceToHost),"output");std::uint32_t bad=0;double maxnorm=0;for(std::uint32_t matrix=0;matrix<o.batch;++matrix)for(std::uint32_t mr=0;mr<o.m;++mr)for(std::uint32_t nr=0;nr<o.n;++nr){double sum=0,ab=0;for(std::uint32_t kk=0;kk<o.k;++kk){auto sc=scales[(nr/128U)*(o.k/128U)+kk/128U];double w=bf16_value(bf16_bits(e4(canon[std::size_t(nr)*o.k+kk])*e8(sc))),av=bf16_value(x[std::size_t(mr)*o.k+kk]);sum+=w*av;ab+=std::abs(w*av);}float expected=bf16_value(bf16_bits(float(sum)));const auto oi=(std::size_t(matrix)*o.m+mr)*o.n+nr;double norm=std::abs(double(out[oi])-expected)/std::max(ab,1e-30);maxnorm=std::max(maxnorm,norm);if(bf16_bits(out[oi])!=bf16_bits(expected))++bad;}
  const std::size_t useful_bytes=std::size_t(o.batch)*matrix_bytes;const double rgb=double(kRulerBytes)/ru/1000.0,kgb=double(useful_bytes)/ku/1000.0,eff=kgb/rgb;std::ostream*os=&std::cout;std::ofstream file;if(!o.output.empty()){file.open(o.output);os=&file;}*os<<std::fixed<<std::setprecision(9)<<"{\n  \"device_name\": \""<<prop.name<<"\",\n  \"M\": "<<o.m<<", \"N\": "<<o.n<<", \"K\": "<<o.k<<", \"batch\": "<<o.batch<<", \"split_k\": "<<o.split<<", \"n_tiles_per_warp\": "<<o.ntiles<<", \"scale_on_b\": "<<(o.scale_b?"true":"false")<<", \"async_copy\": "<<(o.async_copy?"true":"false")<<", \"scale_accumulator\": "<<(o.scale_accumulator?"true":"false")<<",\n  \"matrix_bytes\": "<<matrix_bytes<<", \"useful_bytes\": "<<useful_bytes<<", \"peak_device_bytes\": "<<peak<<",\n  \"ruler_us\": "<<ru<<", \"ruler_gbps\": "<<rgb<<",\n  \"kernel_us\": "<<ku<<", \"per_matrix_us\": "<<(ku/o.batch)<<", \"effective_gbps\": "<<kgb<<", \"roofline_efficiency\": "<<eff<<",\n  \"bf16_mismatches\": "<<bad<<", \"maximum_error_over_sum_abs\": "<<maxnorm<<"\n}\n";
  check(cudaEventDestroy(a),"destroy");check(cudaEventDestroy(b),"destroy");check(cudaStreamDestroy(st),"destroy stream");return bad==0?EXIT_SUCCESS:EXIT_FAILURE;
 }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return EXIT_FAILURE;}
}
