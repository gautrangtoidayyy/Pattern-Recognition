// Seeded feature-permutation logic adapted from scikit-learn tree/_splitter.pyx
// and utils/_random.pxd (BSD-3-Clause); see vendor/scikit-learn-LICENSE.txt.
import {infer} from './engine.js?v=20261104-theory';
const yieldTask=()=>new Promise(r=>setTimeout(r,0));
export function preprocess(bytes,meta,count){const x=new Float32Array(count*784);for(let i=0;i<x.length;i++)x[i]=Math.fround(Math.fround(Math.fround(bytes[i]/255)-meta.mean)/meta.std);return x;}
export function evaluate(net,model,bytes,labels,count){let correct=0,loss=0;for(let i=0;i<count;i++){const raw=Float32Array.from(bytes.subarray(i*784,(i+1)*784),v=>v/255),r=infer(net,model,raw);correct+=r.pred===labels[i];loss-=Math.log(Math.max(1e-12,r.probs[labels[i]]));}return{accuracy:correct/count,loss:loss/count};}
export async function fitBayes({model,bytes,labels,meta,count,smoothing=.1,onUpdate=()=>{},shouldStop=()=>false}){
 const X=preprocess(bytes,meta,count),counts=new Float64Array(10),sum=new Float64Array(7840),square=new Float64Array(7840),globalSum=new Float64Array(784),globalSq=new Float64Array(784);let seen=0,last=null;
 for(let start=0;start<count;start+=128){if(shouldStop())break;const end=Math.min(count,start+128);
  for(let i=start;i<end;i++){const c=labels[i];counts[c]++;seen++;for(let j=0;j<784;j++){const v=model==='gnb'?X[i*784+j]:model==='bnb'?+(bytes[i*784+j]/255>.5):Math.fround(bytes[i*784+j]/255);sum[c*784+j]+=v;square[c*784+j]+=v*v;globalSum[j]+=v;globalSq[j]+=v*v;}}
  const prior=counts.map(n=>Math.max(n/seen,1e-12)),w={};w[model+'.prior']=prior;
  if(model==='gnb'){let maxVar=0;for(let j=0;j<784;j++)maxVar=Math.max(maxVar,globalSq[j]/seen-(globalSum[j]/seen)**2);const mean=new Float64Array(7840),variance=new Float64Array(7840);for(let c=0;c<10;c++)for(let j=0;j<784;j++){const n=Math.max(1,counts[c]),q=c*784+j;mean[q]=sum[q]/n;variance[q]=Math.max(0,square[q]/n-mean[q]**2)+Math.max(1e-9,smoothing*maxVar);}w['gnb.mean']=mean;w['gnb.var']=variance;w['gnb.var_smoothing']=Float64Array.of(smoothing);}
  else if(model==='bnb'){const theta=new Float64Array(7840);for(let c=0;c<10;c++)for(let j=0;j<784;j++)theta[c*784+j]=(sum[c*784+j]+smoothing)/(counts[c]+2*smoothing);w['bnb.theta']=theta;w['bnb.threshold']=Float64Array.of(.5);}
  else{w['mnb.feature_count']=Float64Array.from(sum);w['mnb.alpha']=Float64Array.of(smoothing);}
  last={w,seen,step:Math.ceil(end/128),total:Math.ceil(count/128)};await onUpdate(last);await yieldTask();
 }return last;
}
// Exact candidate CART for MNIST: no quantization. The 256 raw intensity
// groups retain every distinct float32 value and every midpoint sklearn can use.
// Feature permutation follows sklearn's MT19937 -> xorshift seed convention.
export class MT19937 {
 constructor(seed){this.a=new Uint32Array(624);this.a[0]=seed>>>0;for(let i=1;i<624;i++)this.a[i]=(Math.imul(1812433253,this.a[i-1]^(this.a[i-1]>>>30))+i)>>>0;this.i=624;}
 uint(){if(this.i===624){for(let k=0;k<624;k++){const y=(this.a[k]&0x80000000)|(this.a[(k+1)%624]&0x7fffffff);this.a[k]=this.a[(k+397)%624]^(y>>>1)^((y&1)?0x9908b0df:0);}this.i=0;}let y=this.a[this.i++];y^=y>>>11;y^=(y<<7)&0x9d2c5680;y^=(y<<15)&0xefc60000;y^=y>>>18;return y>>>0;}
 int31(){let x;do{x=this.uint()&0x7fffffff;}while(x===2147483647);return x;}
}
export function prepareExactPixels(bytes,count,meta){
 const values=new Float32Array(256);for(let v=0;v<256;v++)values[v]=Math.fround(Math.fround(Math.fround(v/255)-meta.mean)/meta.std);
 const columns=new Uint8Array(count*784);for(let i=0;i<count;i++)for(let f=0;f<784;f++)columns[f*count+i]=bytes[i*784+f];
 return {columns,values,count};
}
export function growExactTree(pixels,y,weights,treeSeed=42,maxDepth=3){
 const {columns,values,count}=pixels,features=Int32Array.from({length:784},(_,i)=>i),constants=new Int32Array(784);
 let random=new MT19937(treeSeed).int31();const rand=(low,high)=>{if(!random)random=1;random^=random<<13;random^=random>>>17;random^=random<<5;return low+((random>>>0)&0x7fffffff)%(high-low);};
 const swap=(a,i,j)=>{const t=a[i];a[i]=a[j];a[j]=t;};
 const tree={left:[],right:[],feature:[],threshold:[],pred:[],seed:treeSeed};
 function build(indices,depth,known){
  const id=tree.left.length;tree.left.push(-1);tree.right.push(-1);tree.feature.push(-2);tree.threshold.push(-2);
  const cls=new Float64Array(10);let total=0;for(const i of indices){cls[y[i]]+=weights[i];total+=weights[i];}
  let pred=0;for(let c=1;c<10;c++)if(cls[c]>cls[pred])pred=c;tree.pred.push(pred);
  let square=0;for(let c=0;c<10;c++)square+=cls[c]*cls[c];const impurity=1-square/(total*total);
  if(depth>=maxDepth||indices.length<2||impurity<=Number.EPSILON)return id;
  let best=-Infinity,bf=-1,blo=-1,bhi=-1,fi=784,found=0,drawn=0,totalConstants=known;
  const hist=new Float64Array(2560),numbers=new Int32Array(256),left=new Float64Array(10);
  while(fi>totalConstants){
   let fj=rand(drawn,fi-found);
   if(fj<known){swap(features,drawn,fj);drawn++;continue;}
   fj+=found;const f=features[fj],offset=f*count;hist.fill(0);numbers.fill(0);
   let lo=255,hi=0;for(const i of indices){const v=columns[offset+i];hist[v*10+y[i]]+=weights[i];numbers[v]++;if(v<lo)lo=v;if(v>hi)hi=v;}
   if(values[hi]<=values[lo]+1e-7){swap(features,fj,totalConstants);found++;totalConstants++;continue;}
   fi--;swap(features,fi,fj);left.fill(0);let lw=0,nl=0;
   for(let v=lo;v<hi;v++){
    if(!numbers[v])continue;
    let next=v+1;while(next<=hi&&!numbers[next])next++;
    if(next>hi)break;
    for(let c=0;c<10;c++){const mass=hist[v*10+c];left[c]+=mass;lw+=mass;}
    nl+=numbers[v];const rw=total-lw;
    if(nl<1||indices.length-nl<1||lw<=0||rw<=0)continue;
    let ls=0,rs=0;for(let c=0;c<10;c++){ls+=left[c]*left[c];const r=cls[c]-left[c];rs+=r*r;}
    const proxy=ls/lw+rs/rw;
    if(proxy>best){best=proxy;bf=f;blo=v;bhi=next;}
   }
  }
  for(let i=0;i<known;i++)features[i]=constants[i];
  for(let i=known;i<totalConstants;i++)constants[i]=features[i];
  if(bf<0)return id;
  const threshold=values[blo]/2+values[bhi]/2,l=[],r=[];
  for(const i of indices)(columns[bf*count+i]<=blo?l:r).push(i);
  if(!l.length||!r.length)return id;
  tree.feature[id]=bf;tree.threshold[id]=threshold;
  tree.left[id]=build(l,depth+1,totalConstants);tree.right[id]=build(r,depth+1,totalConstants);return id;
 }
 build(Array.from({length:count},(_,i)=>i).filter(i=>weights[i]>0),0,0);return tree;
}
// Compatibility entry for callers with normalized float32 MNIST inputs.
export function growTree(X,y,weights,count,bins=256,maxDepth=3){
 const values=Float32Array.from(new Set(X)).sort(),columns=new Uint8Array(count*784),lookup=new Map(Array.from(values,(v,i)=>[v,i]));
 if(values.length>256)throw Error('Exact MNIST CART expects at most 256 pixel intensities.');
 for(let i=0;i<count;i++)for(let f=0;f<784;f++)columns[f*count+i]=lookup.get(X[i*784+f]);
 return growExactTree({columns,values,count},y,weights,42,maxDepth);
}
export function treePredict(tree,x,offset=0){let n=0;while(tree.left[n]!==-1)n=x[offset+tree.feature[n]]<=tree.threshold[n]?tree.left[n]:tree.right[n];return tree.pred[n];}
export async function fitAda({bytes,labels,meta,count,rounds=200,learningRate=.5,seed=42,onUpdate=()=>{},onStatus=()=>{},shouldStop=()=>false}){
 const pixels=prepareExactPixels(bytes,count,meta),sampleWeights=new Float64Array(count).fill(1/count),trees=[],weights=[],errors=[],rng=new MT19937(seed);let last=null;
 const predict=(tree,i)=>{let n=0;while(tree.left[n]!==-1)n=pixels.values[bytes[i*784+tree.feature[n]]]<=tree.threshold[n]?tree.left[n]:tree.right[n];return tree.pred[n];};
 for(let round=0;round<rounds;round++){
  if(shouldStop())break;
  onStatus({round:round+1,total:rounds,count});await yieldTask();if(shouldStop())break;
  // sklearn clips positive weights to float64 epsilon before each weak learner.
  for(let i=0;i<count;i++)if(sampleWeights[i]>0)sampleWeights[i]=Math.max(Number.EPSILON,sampleWeights[i]);
  const tree=growExactTree(pixels,labels,sampleWeights,rng.int31()),preds=new Uint8Array(count);let error=0,mass=0;
  for(let i=0;i<count;i++){mass+=sampleWeights[i];preds[i]=predict(tree,i);if(preds[i]!==labels[i])error+=sampleWeights[i];}error/=mass;
  if(error>=.9)break;
  const alpha=error<=0?1:learningRate*(Math.log((1-error)/error)+Math.log(9));
  trees.push(tree);weights.push(alpha);errors.push(error);
  last={trees:[...trees],weights:[...weights],errors:[...errors],step:round+1,total:rounds,seen:count,seed,splitter:'all-distinct-midpoints',maxDepth:3};
  await onUpdate(last);await yieldTask();if(error<=0)break;
  let total=0;for(let i=0;i<count;i++){if(sampleWeights[i]>0)sampleWeights[i]=Math.exp(Math.log(sampleWeights[i])+(preds[i]!==labels[i]?alpha:0));total+=sampleWeights[i];}
  for(let i=0;i<count;i++)sampleWeights[i]/=total;
 }
 if(!last&&!shouldStop())throw Error('Không tìm được cây có lỗi nhỏ hơn 90%. Hãy tăng số mẫu train.');return last;
}
