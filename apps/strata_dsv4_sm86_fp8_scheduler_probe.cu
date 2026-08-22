// Ampere-native layer-resident FP8 projection scheduler probe.
// Compact E4M3/E8M0 weights remain byte-resident. One persistent launch runs
// q_a -> {q_b + indexer, wkv} -> wo_a -> wo_b with real BF16 dependencies.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
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
constexpr std::uint32_t kWarp=32,kWarps=4,kThreads=128,kResidentBlocksPerSm=6;
constexpr std::uint32_t kWarmups=3,kSamples=11;
constexpr std::size_t kMiB=1ULL<<20,kScrubBytes=256ULL*kMiB,
                      kRulerBytes=128ULL*kMiB,kCeiling=512ULL*kMiB;

void check(cudaError_t s,std::string_view w){if(s!=cudaSuccess)throw std::runtime_error(std::string(w)+": "+cudaGetErrorString(s));}
class Buffer{public:explicit Buffer(std::size_t n):n_(n){check(cudaMalloc(&p_,n),"cudaMalloc scheduler");}~Buffer(){if(p_)cudaFree(p_);}void*get()const{return p_;}std::size_t bytes()const{return n_;}private:void*p_{};std::size_t n_{};};
std::uint32_t rng(std::uint32_t&x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
std::uint16_t bf16_bits(float x){auto b=std::bit_cast<std::uint32_t>(x);b+=0x7fffU+((b>>16)&1U);return std::uint16_t(b>>16);}
float bf16_value(std::uint16_t b){return std::bit_cast<float>(std::uint32_t(b)<<16);}
float e4(std::uint8_t c){auto e=(c>>3)&15U,m=c&7U;float v=e?std::ldexp(1.0F+float(m)/8.0F,int(e)-7):std::ldexp(float(m),-9);return(c&0x80U)?-v:v;}
float e8(std::uint8_t s){return std::ldexp(1.0F,int(s)-127);}

__device__ __forceinline__ std::uint32_t decode_pair(std::uint32_t pair,std::uint32_t factor){const auto p=__byte_perm(pair,0U,0x4140U);const std::uint32_t v=((p<<8)&0x80008000U)|((p<<4)&0x07f007f0U);const auto r=__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&v),*reinterpret_cast<const __nv_bfloat162*>(&factor));return *reinterpret_cast<const std::uint32_t*>(&r);}
__device__ __forceinline__ void mma(float&d0,float&d1,float&d2,float&d3,std::uint32_t a0,std::uint32_t a1,std::uint32_t a2,std::uint32_t a3,std::uint32_t b0,std::uint32_t b1){asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n":"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));}
__device__ __forceinline__ std::uint16_t round_bf16(float x){std::uint32_t b=__float_as_uint(x);b+=0x7fffU+((b>>16)&1U);return std::uint16_t(b>>16);}

__device__ __forceinline__ float device_bf16(std::uint16_t bits){return __uint_as_float(std::uint32_t(bits)<<16U);}

template<int SPLIT,bool GROUPED=false>
__device__ __forceinline__ void process_work(const uint4*codes,const std::uint8_t*scales,const std::uint16_t*x,float*partials,std::uint32_t*counters,std::uint16_t*y,std::uint32_t n,std::uint32_t k,std::uint32_t work,std::uint32_t*arrived,std::uint32_t*replays){
 const std::uint32_t lane=threadIdx.x&31U,warp=threadIdx.x>>5U,nt=work/SPLIT,slice=work%SPLIT,kpairs=k/32U,pairs_per_slice=kpairs/SPLIT,pbegin=slice*pairs_per_slice,pend=pbegin+pairs_per_slice,group=lane>>2U,thread=lane&3U;
 x+=GROUPED?std::size_t(nt/64U)*k:0U;
 float e0=0,e1=0,e2=0,e3=0,o0=0,o1=0,o2=0,o3=0;
 for(std::uint32_t pb=pbegin;pb<pend;pb+=4U){const std::uint32_t ktb=pb*2U;std::uint32_t s=lane==0?scales[(nt/8U)*(k/128U)+ktb/8U]:0U;s=__shfl_sync(0xffffffffU,s,0);const std::uint32_t factor=((s+120U)<<7U)*0x00010001U;uint4 q[4];
#pragma unroll
  for(int u=0;u<4;++u)q[u]=codes[(std::size_t(nt)*kpairs+pb+std::uint32_t(u))*kWarp+lane];
#pragma unroll
  for(int u=0;u<4;++u)
#pragma unroll
   for(int j=0;j<2;++j){const std::uint32_t lo=j==0?q[u].x:q[u].z,hi=j==0?q[u].y:q[u].w;const auto a0=decode_pair(lo&0xffffU,factor),a1=decode_pair(lo>>16,factor),a2=decode_pair(hi&0xffffU,factor),a3=decode_pair(hi>>16,factor);std::uint32_t b0=0,b1=0;if(group==0U){const std::uint32_t kt=ktb+std::uint32_t(u)*2U+std::uint32_t(j);const auto*base=x+kt*16U+thread*2U;b0=*reinterpret_cast<const std::uint32_t*>(base);b1=*reinterpret_cast<const std::uint32_t*>(base+8U);}if(j==0)mma(e0,e1,e2,e3,a0,a1,a2,a3,b0,b1);else mma(o0,o1,o2,o3,a0,a1,a2,a3,b0,b1);}
 }
 e0+=o0;e1+=o1;e2+=o2;e3+=o3;float*dst=partials+(std::size_t(nt)*SPLIT+slice)*128U+lane*4U;dst[0]=e0;dst[1]=e1;dst[2]=e2;dst[3]=e3;__threadfence();if(lane==0)arrived[warp]=atomicAdd(counters+nt,1U);__syncwarp();if(arrived[warp]!=SPLIT-1U)return;
 if(lane<16U){const std::uint32_t nr=lane,source=(nr&7U)*4U,reg=(nr>>3U)*2U;float sum=0.0F;
#pragma unroll
  for(std::uint32_t sk=0;sk<SPLIT;++sk)sum+=partials[(std::size_t(nt)*SPLIT+sk)*128U+source*4U+reg];y[nt*16U+nr]=round_bf16(sum);}
 (void)replays;
 if(lane==0)counters[nt]=0U;
}

__device__ void grid_barrier(std::uint32_t*count,std::uint32_t*epoch){__shared__ std::uint32_t seen;if(threadIdx.x==0){seen=atomicAdd(epoch,0U);__threadfence();const auto ticket=atomicAdd(count,1U);if(ticket==gridDim.x-1U){atomicExch(count,0U);__threadfence();atomicAdd(epoch,1U);}else while(atomicAdd(epoch,0U)==seen){}}__syncthreads();}

__device__ void grid_barrier_qnorm(std::uint32_t*count,std::uint32_t*epoch,const std::uint16_t*source,const std::uint16_t*weight,std::uint16_t*output){__shared__ std::uint32_t seen,last;__shared__ float inverse;if(threadIdx.x==0){seen=atomicAdd(epoch,0U);__threadfence();last=atomicAdd(count,1U)==gridDim.x-1U;}__syncthreads();if(last!=0U){float sum=0.0F;if(threadIdx.x<kWarp){for(std::uint32_t i=threadIdx.x;i<1024U;i+=kWarp){const float value=device_bf16(source[i]);sum+=value*value;}for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffffU,sum,offset);if(threadIdx.x==0)inverse=1.0F/sqrtf(sum/1024.0F+1.0e-6F);}__syncthreads();for(std::uint32_t i=threadIdx.x;i<1024U;i+=kThreads)output[i]=round_bf16(device_bf16(source[i])*inverse*device_bf16(weight[i]));__syncthreads();if(threadIdx.x==0){atomicExch(count,0U);__threadfence();atomicAdd(epoch,1U);}}else if(threadIdx.x==0)while(atomicAdd(epoch,0U)==seen){}__syncthreads();}

__global__ __launch_bounds__(kThreads,6) void scheduler_kernel(
 const uint4*qa,const std::uint8_t*qas,const uint4*qb,const std::uint8_t*qbs,const uint4*kv,const std::uint8_t*kvs,const uint4*oa,const std::uint8_t*oas,const uint4*ob,const std::uint8_t*obs,
 const std::uint16_t*input,const std::uint16_t*attention,const std::uint16_t*qnorm,float*pqa,float*pqb,float*pkv,float*poa,float*pob,std::uint32_t*cqa,std::uint32_t*cqb,std::uint32_t*ckv,std::uint32_t*coa,std::uint32_t*cob,
 std::uint16_t*yqa,std::uint16_t*yqr,std::uint16_t*yqb,std::uint16_t*ykv,std::uint16_t*yoa,std::uint16_t*yob,std::uint32_t*barrier,std::uint32_t*replays){
 __shared__ std::uint32_t arrived[kWarps];const std::uint32_t pw=blockIdx.x*kWarps+(threadIdx.x>>5U),stride=gridDim.x*kWarps;
 for(std::uint32_t w=pw;w<(1024U/16U)*16U;w+=stride)process_work<16>(qa,qas,input,pqa,cqa,yqa,1024,4096,w,arrived,replays);
 grid_barrier_qnorm(barrier,barrier+1,yqa,qnorm,yqr);
 constexpr std::uint32_t qbwork=(40960U/16U),kvwork=(512U/16U)*16U;
 for(std::uint32_t w=pw;w<qbwork+kvwork;w+=stride){if(w<qbwork)process_work<1>(qb,qbs,yqr,pqb,cqb,yqb,40960,1024,w,arrived,replays+1);else process_work<16>(kv,kvs,input,pkv,ckv,ykv,512,4096,w-qbwork,arrived,replays+2);}
 grid_barrier(barrier,barrier+1);
 for(std::uint32_t w=pw;w<(8192U/16U)*4U;w+=stride)process_work<4,true>(oa,oas,attention,poa,coa,yoa,8192,4096,w,arrived,replays+3);
 grid_barrier(barrier,barrier+1);
 for(std::uint32_t w=pw;w<(4096U/16U)*8U;w+=stride)process_work<8>(ob,obs,yoa,pob,cob,yob,4096,8192,w,arrived,replays+4);
}

__global__ void scrub_kernel(std::uint8_t*p,std::size_t n){for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x,s=std::size_t(gridDim.x)*blockDim.x;i<n;i+=s)p[i]=std::uint8_t(i);}
__global__ void ruler_kernel(const uint4*p,std::uint64_t n,unsigned*sink){const std::uint64_t st=std::uint64_t(gridDim.x)*blockDim.x;std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;unsigned a=0;for(;i<n;i+=st){auto x=p[i];a^=x.x^x.y^x.z^x.w;}if(a==0xdeadbeefU)sink[threadIdx.x]=a;}
float median(std::vector<float>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
template<class F>float timed(cudaEvent_t a,cudaEvent_t b,cudaStream_t s,F f){check(cudaEventRecord(a,s),"start");f();check(cudaEventRecord(b,s),"stop");check(cudaEventSynchronize(b),"sync");float ms;check(cudaEventElapsedTime(&ms,a,b),"elapsed");return ms*1000.0F;}

struct HostMatrix{std::uint32_t n,k;std::vector<std::uint8_t>canon,scales;std::vector<uint4>packed;HostMatrix(std::uint32_t nn,std::uint32_t kk,std::uint32_t&state):n(nn),k(kk),canon(std::size_t(n)*k),scales(std::size_t(n/128U)*(k/128U)),packed(std::size_t(n/16U)*(k/32U)*32U){for(auto&c:canon){c=std::uint8_t(rng(state));if((c&0x7fU)==0x7fU)c^=1U;}for(auto&s:scales)s=std::uint8_t(114U+rng(state)%7U);for(std::uint32_t nt=0;nt<n/16U;++nt)for(std::uint32_t kp=0;kp<k/32U;++kp)for(std::uint32_t lane=0;lane<32;++lane){std::uint32_t words[4]{};for(std::uint32_t j=0;j<2;++j)for(std::uint32_t i=0;i<8;++i){const bool high=i==2||i==3||i==6||i==7;const auto row=nt*16+(lane>>2)+(high?8:0),col=(kp*2+j)*16+(lane&3)*2+(i&1)+(i>=4?8:0);words[j*2+(i>=4)]|=std::uint32_t(canon[std::size_t(row)*k+col])<<((i&3)*8);}packed[(std::size_t(nt)*(k/32U)+kp)*32+lane]=make_uint4(words[0],words[1],words[2],words[3]);}}std::size_t bytes()const{return canon.size()+scales.size();}};
struct DevMatrix{Buffer code,scale,partial,counter,out;DevMatrix(const HostMatrix&h,std::uint32_t split):code(h.canon.size()),scale(h.scales.size()),partial(std::size_t(h.n/16U)*split*128U*sizeof(float)),counter(std::size_t(h.n/16U)*sizeof(std::uint32_t)),out(std::size_t(h.n)*sizeof(std::uint16_t)){}std::size_t bytes()const{return code.bytes()+scale.bytes()+partial.bytes()+counter.bytes()+out.bytes();}};
struct Error{std::uint32_t bad{};double norm{};};
Error oracle(const HostMatrix&w,const std::vector<std::uint16_t>&x,const std::vector<std::uint16_t>&y,bool grouped=false){Error r;for(std::uint32_t n=0;n<w.n;++n){double sum=0,ab=0;const auto base=grouped?std::size_t(n/1024U)*w.k:0U;for(std::uint32_t k=0;k<w.k;++k){const auto s=w.scales[(n/128U)*(w.k/128U)+k/128U];const double p=double(bf16_value(bf16_bits(e4(w.canon[std::size_t(n)*w.k+k])*e8(s))))*bf16_value(x[base+k]);sum+=p;ab+=std::abs(p);}const auto expected=bf16_bits(float(sum));if(y[n]!=expected)++r.bad;r.norm=std::max(r.norm,std::abs(double(bf16_value(y[n]))-bf16_value(expected))/std::max(ab,1e-30));}return r;}
std::vector<std::uint16_t> normalize(const std::vector<std::uint16_t>&x,const std::vector<std::uint16_t>&weight){float partial[kWarp]{};for(std::uint32_t thread=0;thread<kWarp;++thread)for(std::uint32_t i=thread;i<1024U;i+=kWarp){const float value=bf16_value(x[i]);partial[thread]+=value*value;}for(std::uint32_t stride=kWarp/2U;stride!=0U;stride>>=1U)for(std::uint32_t i=0;i<stride;++i)partial[i]+=partial[i+stride];const float inverse=1.0F/std::sqrt(partial[0]/1024.0F+1.0e-6F);std::vector<std::uint16_t>result(1024U);for(std::uint32_t i=0;i<1024U;++i)result[i]=bf16_bits(bf16_value(x[i])*inverse*bf16_value(weight[i]));return result;}
Error compare_codes(const std::vector<std::uint16_t>&expected,const std::vector<std::uint16_t>&actual){Error result;for(std::size_t i=0;i<expected.size();++i){if(expected[i]!=actual[i])++result.bad;result.norm=std::max(result.norm,std::abs(double(bf16_value(expected[i]))-bf16_value(actual[i])));}return result;}
}

int main(int argc,char**argv){try{
 std::string output;for(int i=1;i<argc;++i){std::string_view f=argv[i];if(f=="--output"&&++i<argc)output=argv[i];else throw std::runtime_error("usage: --output FILE");}
 check(cudaSetDevice(0),"device");cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0),"properties");if(prop.major!=8||prop.minor!=6)throw std::runtime_error("requires SM86");
 std::uint32_t state=0x514a9c31U;HostMatrix qa(1024,4096,state),qb(40960,1024,state),kv(512,4096,state),oa(8192,4096,state),ob(4096,8192,state);DevMatrix dqa(qa,16),dqb(qb,1),dkv(kv,16),doa(oa,4),dob(ob,8);
 std::vector<std::uint16_t>input(4096),attention(32768),qnorm(1024,bf16_bits(1.0F));for(auto&v:input)v=bf16_bits(float(int(rng(state)%17)-8)/8.0F);for(auto&v:attention)v=bf16_bits(float(int(rng(state)%17)-8)/8.0F);
 Buffer din(input.size()*2),datt(attention.size()*2),dnorm(qnorm.size()*2),yqr(qnorm.size()*2),barrier(8),replays(5*sizeof(std::uint32_t)),scrub(kScrubBytes),ruler(kRulerBytes),sink(1024);
 const std::size_t peak=dqa.bytes()+dqb.bytes()+dkv.bytes()+doa.bytes()+dob.bytes()+din.bytes()+datt.bytes()+dnorm.bytes()+yqr.bytes()+barrier.bytes()+replays.bytes()+scrub.bytes()+ruler.bytes()+sink.bytes();if(peak>kCeiling)throw std::runtime_error("exceeds 512 MiB");
 cudaStream_t st;check(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"stream");auto upload=[&](const HostMatrix&h,DevMatrix&d){check(cudaMemcpyAsync(d.code.get(),h.packed.data(),d.code.bytes(),cudaMemcpyHostToDevice,st),"code");check(cudaMemcpyAsync(d.scale.get(),h.scales.data(),d.scale.bytes(),cudaMemcpyHostToDevice,st),"scale");check(cudaMemsetAsync(d.counter.get(),0,d.counter.bytes(),st),"counter");};upload(qa,dqa);upload(qb,dqb);upload(kv,dkv);upload(oa,doa);upload(ob,dob);
 check(cudaMemcpyAsync(din.get(),input.data(),din.bytes(),cudaMemcpyHostToDevice,st),"input");check(cudaMemcpyAsync(datt.get(),attention.data(),datt.bytes(),cudaMemcpyHostToDevice,st),"attention");check(cudaMemcpyAsync(dnorm.get(),qnorm.data(),dnorm.bytes(),cudaMemcpyHostToDevice,st),"qnorm");check(cudaMemsetAsync(barrier.get(),0,barrier.bytes(),st),"barrier");check(cudaMemsetAsync(replays.get(),0,replays.bytes(),st),"replays");check(cudaMemsetAsync(ruler.get(),0xa5,ruler.bytes(),st),"ruler");check(cudaStreamSynchronize(st),"uploads");
 const auto blocks=prop.multiProcessorCount*kResidentBlocksPerSm;auto run=[&]{scheduler_kernel<<<blocks,kThreads,0,st>>>((const uint4*)dqa.code.get(),(const std::uint8_t*)dqa.scale.get(),(const uint4*)dqb.code.get(),(const std::uint8_t*)dqb.scale.get(),(const uint4*)dkv.code.get(),(const std::uint8_t*)dkv.scale.get(),(const uint4*)doa.code.get(),(const std::uint8_t*)doa.scale.get(),(const uint4*)dob.code.get(),(const std::uint8_t*)dob.scale.get(),(const std::uint16_t*)din.get(),(const std::uint16_t*)datt.get(),(const std::uint16_t*)dnorm.get(),(float*)dqa.partial.get(),(float*)dqb.partial.get(),(float*)dkv.partial.get(),(float*)doa.partial.get(),(float*)dob.partial.get(),(std::uint32_t*)dqa.counter.get(),(std::uint32_t*)dqb.counter.get(),(std::uint32_t*)dkv.counter.get(),(std::uint32_t*)doa.counter.get(),(std::uint32_t*)dob.counter.get(),(std::uint16_t*)dqa.out.get(),(std::uint16_t*)yqr.get(),(std::uint16_t*)dqb.out.get(),(std::uint16_t*)dkv.out.get(),(std::uint16_t*)doa.out.get(),(std::uint16_t*)dob.out.get(),(std::uint32_t*)barrier.get(),(std::uint32_t*)replays.get());};
 auto doscrub=[&]{scrub_kernel<<<65536,256,0,st>>>((std::uint8_t*)scrub.get(),scrub.bytes());};auto doruler=[&]{ruler_kernel<<<prop.multiProcessorCount*8,256,0,st>>>((const uint4*)ruler.get(),ruler.bytes()/16,(unsigned*)sink.get());};for(unsigned i=0;i<kWarmups;++i){doruler();run();}check(cudaStreamSynchronize(st),"warmups");cudaEvent_t a,b;check(cudaEventCreate(&a),"event");check(cudaEventCreate(&b),"event");std::vector<float>rt,kt;for(unsigned i=0;i<kSamples;++i){doscrub();rt.push_back(timed(a,b,st,doruler));doscrub();kt.push_back(timed(a,b,st,run));}check(cudaGetLastError(),"scheduler");
 auto download=[&](DevMatrix&d,const HostMatrix&h){std::vector<std::uint16_t>v(h.n);check(cudaMemcpy(v.data(),d.out.get(),d.out.bytes(),cudaMemcpyDeviceToHost),"output");return v;};auto yqa=download(dqa,qa),yqb=download(dqb,qb),ykv=download(dkv,kv),yoa=download(doa,oa),yob=download(dob,ob);std::vector<std::uint16_t>yqr_host(1024);check(cudaMemcpy(yqr_host.data(),yqr.get(),yqr.bytes(),cudaMemcpyDeviceToHost),"query norm output");std::uint32_t replay_total[5]{};check(cudaMemcpy(replay_total,replays.get(),replays.bytes(),cudaMemcpyDeviceToHost),"replay counters");
 const auto expected_qr=normalize(yqa,qnorm);const auto eqa=oracle(qa,input,yqa),eqr=compare_codes(expected_qr,yqr_host),eqb=oracle(qb,yqr_host,yqb),ekv=oracle(kv,input,ykv),eoa=oracle(oa,attention,yoa,true),eob=oracle(ob,yoa,yob);
 const float ru=median(rt),ku=median(kt);const std::size_t useful=qa.bytes()+qb.bytes()+kv.bytes()+oa.bytes()+ob.bytes();const double rgb=double(kRulerBytes)/ru/1000.0,gb=double(useful)/ku/1000.0;constexpr std::uint32_t runs=kWarmups+kSamples;const std::uint64_t replay_bytes=std::uint64_t(replay_total[0]/runs)*(4096U+32U)+std::uint64_t(replay_total[1]/runs)*(1024U+8U)+std::uint64_t(replay_total[2]/runs)*(4096U+32U)+std::uint64_t(replay_total[3]/runs)*(4096U+32U)+std::uint64_t(replay_total[4]/runs)*(8192U+64U);
 std::ostream*os=&std::cout;std::ofstream file;if(!output.empty()){file.open(output);os=&file;}*os<<std::fixed<<std::setprecision(9)<<"{\n  \"device_name\": \""<<prop.name<<"\", \"resident_blocks\": "<<blocks<<",\n  \"useful_bytes\": "<<useful<<", \"replay_weight_bytes\": "<<replay_bytes<<", \"peak_device_bytes\": "<<peak<<",\n  \"ruler_us\": "<<ru<<", \"ruler_gbps\": "<<rgb<<",\n  \"scheduler_us\": "<<ku<<", \"effective_gbps\": "<<gb<<", \"roofline_efficiency\": "<<(gb/rgb)<<",\n  \"replay_rows\": ["<<replay_total[0]/runs<<","<<replay_total[1]/runs<<","<<replay_total[2]/runs<<","<<replay_total[3]/runs<<","<<replay_total[4]/runs<<"],\n  \"q_a_mismatches\": "<<eqa.bad<<", \"q_a_maximum_error_over_sum_abs\": "<<eqa.norm<<",\n  \"q_norm_mismatches\": "<<eqr.bad<<", \"q_norm_maximum_absolute\": "<<eqr.norm<<",\n  \"q_b_indexer_mismatches\": "<<eqb.bad<<", \"q_b_indexer_maximum_error_over_sum_abs\": "<<eqb.norm<<",\n  \"wkv_mismatches\": "<<ekv.bad<<", \"wkv_maximum_error_over_sum_abs\": "<<ekv.norm<<",\n  \"wo_a_mismatches\": "<<eoa.bad<<", \"wo_a_maximum_error_over_sum_abs\": "<<eoa.norm<<",\n  \"wo_b_mismatches\": "<<eob.bad<<", \"wo_b_maximum_error_over_sum_abs\": "<<eob.norm<<"\n}\n";cudaEventDestroy(a);cudaEventDestroy(b);cudaStreamDestroy(st);return EXIT_SUCCESS;
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return EXIT_FAILURE;}}
