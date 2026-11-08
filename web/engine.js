export const names={bnb:'Naïve Bayes · Bernoulli',gnb:'Naïve Bayes · Gaussian',mnb:'Naïve Bayes · Multinomial',lr:'Logistic Regression',mlp:'MLP',cnn:'CNN',ada:'AdaBoost'};
export function hydrate(meta,buffer,trees){const values=new Float64Array(buffer);const w={};for(const[k,v]of Object.entries(meta.arrays))w[k]=values.subarray(v.offset,v.offset+v.length);return{meta,w,...trees};}
export function softmax(x){const m=Math.max(...x),e=Float64Array.from(x,v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);}
function linear(x,w,b){const out=Float64Array.from(b);for(let o=0;o<b.length;o++){let s=b[o];for(let i=0;i<x.length;i++)s+=x[i]*w[o*x.length+i];out[o]=s;}return out;}
const relu=x=>Float64Array.from(x,v=>Math.max(0,v));
export function infer(net,model,raw,exactInput=null){
 const {w,meta}=net,cnnMinmax=model==='cnn'&&meta.models.cnn.preprocessing==='minmax',x=exactInput||Float32Array.from(raw,v=>cnnMinmax?Math.fround(v):Math.fround(Math.fround(Math.fround(v)-meta.mean)/meta.std));
 const binary=model==='bnb'?Float64Array.from(raw,v=>v>(w['bnb.threshold']?.[0]??.5)?1:0):null;
 const stages=[],stage=(name,data,shape,op,extra={})=>{stages.push({name,data,shape,op,...extra});return data;};
 stage('Ảnh gốc',Float64Array.from(raw),[1,28,28],'raw');
 stage(model==='bnb'?'Nhị phân · ngưỡng 0,5':model==='mnb'||cnnMinmax?'Min–max [0, 1]':'Z-score',model==='bnb'?binary:model==='mnb'?raw:x,[1,28,28],'normalize');
 if(model!=='cnn')stage('Flatten · 784',model==='bnb'?binary:model==='mnb'?raw:x,[784],'flatten');
 let logits,probs;
 if(model==='gnb'||model==='mnb'||model==='bnb'){
  const all=new Float64Array(7840),prior=new Float64Array(10),score=new Float64Array(10);
  for(let c=0;c<10;c++){
   prior[c]=Math.log(w[model+'.prior'][c]);let sum=prior[c],den=0;
   if(model==='mnb')for(let i=0;i<784;i++)den+=w['mnb.feature_count'][c*784+i]+w['mnb.alpha'][0];
   for(let i=0;i<784;i++){
    let v;
    if(model==='gnb'){const mu=w['gnb.mean'][c*784+i],variance=w['gnb.var'][c*784+i];v=-.5*Math.log(2*Math.PI*variance)-(x[i]-mu)**2/(2*variance);}
    else if(model==='bnb'){const theta=w['bnb.theta'][c*784+i];v=binary[i]?Math.log(theta):Math.log1p(-theta);}
    else v=raw[i]*Math.log((w['mnb.feature_count'][c*784+i]+w['mnb.alpha'][0])/den);
    all[c*784+i]=v;sum+=v;
   }score[c]=sum;
  }
  stage('Log-likelihood từng pixel',all,[10,28,28],'bayes',{input:model==='gnb'?x:model==='bnb'?binary:raw,model});
  stage('Log-prior',prior,[10],'prior');logits=stage('Log-posterior',score,[10],'posterior');probs=softmax(logits);
 }else if(model==='lr'){
  const W=w['lr.W'],terms=new Float64Array(7840),s=new Float64Array(10);
  for(let c=0;c<10;c++){s[c]=W[784*10+c];for(let i=0;i<784;i++){const v=x[i]*W[i*10+c];terms[c*784+i]=v;s[c]+=v;}}
  stage('Đóng góp xᵢ × wᵢ',terms,[10,28,28],'lrterms',{input:x});logits=stage('Logits · Wx + b',s,[10],'lrlogits');probs=softmax(logits);
 }else if(model==='mlp'){
  let a=x;for(const key of ['0','2','4']){
   const inp=a,b=w['mlp.'+key+'.bias'],W=w['mlp.'+key+'.weight'];a=stage('Dense '+b.length,linear(a,W,b),[b.length],'linear',{input:inp,weights:W,bias:b});
   if(key!=='4')a=stage('ReLU '+b.length,relu(a),[b.length],'relu',{input:a});
  }logits=a;probs=softmax(a);
 }else if(model==='cnn'){
  function conv(inp,cin,side,cout,W,b,name){const out=new Float64Array(cout*side*side);for(let c=0;c<cout;c++)for(let y=0;y<side;y++)for(let x=0;x<side;x++){let sum=b[c];for(let ic=0;ic<cin;ic++)for(let ky=0;ky<3;ky++)for(let kx=0;kx<3;kx++){const iy=y+ky-1,ix=x+kx-1;if(iy>=0&&iy<side&&ix>=0&&ix<side)sum+=inp[ic*side*side+iy*side+ix]*W[(c*cin+ic)*9+ky*3+kx];}out[c*side*side+y*side+x]=sum;}return stage(name,out,[cout,side,side],'conv',{input:inp,inputShape:[cin,side,side],weights:W,bias:b,kernelSize:3,padding:1});}
  function pool(inp,channels,side,name){const h=side/2,out=new Float64Array(channels*h*h);for(let c=0;c<channels;c++)for(let y=0;y<h;y++)for(let x=0;x<h;x++){let i=c*side*side+y*2*side+x*2;out[c*h*h+y*h+x]=Math.max(inp[i],inp[i+1],inp[i+side],inp[i+side+1]);}return stage(name,out,[channels,h,h],'pool',{input:inp,inputShape:[channels,side,side]});}
  let z1=conv(x,1,28,8,w['cnn.c1.weight'],w['cnn.c1.bias'],'Conv 1 · 8 × 3×3');let a1=stage('ReLU 1',relu(z1),[8,28,28],'relu',{input:z1});let p1=pool(a1,8,28,'MaxPool 1');
  let z2=conv(p1,8,14,16,w['cnn.c2.weight'],w['cnn.c2.bias'],'Conv 2 · 16 × 3×3');let a2=stage('ReLU 2',relu(z2),[16,14,14],'relu',{input:z2});let p2=pool(a2,16,14,'MaxPool 2');stage('Flatten · CHW',p2,[784],'flatten');
  let h=stage('Dense 64',linear(p2,w['cnn.f1.weight'],w['cnn.f1.bias']),[64],'linear',{input:p2,weights:w['cnn.f1.weight'],bias:w['cnn.f1.bias']});let act=stage('ReLU 64',relu(h),[64],'relu',{input:h});logits=stage('Dense 10',linear(act,w['cnn.f2.weight'],w['cnn.f2.bias']),[10],'linear',{input:act,weights:w['cnn.f2.weight'],bias:w['cnn.f2.bias']});probs=softmax(logits);
 }else if(model==='ada'){
  const votes=new Float64Array(net.trees.length),accum=new Float64Array(net.trees.length*10),s=new Float64Array(10),paths=[];let total=net.weights.reduce((a,b)=>a+b,0);
  net.trees.forEach((t,i)=>{let n=0,path=[];while(t.left[n]!==-1){const f=t.feature[n],go=x[f]<=t.threshold[n];path.push({node:n,feature:f,value:x[f],threshold:t.threshold[n],left:go});n=go?t.left[n]:t.right[n];}const pred=t.pred[n];path.push({node:n,pred});paths.push(path);votes[i]=pred;
   for(let c=0;c<10;c++){s[c]+=net.weights[i]*(pred===c?1:-1/9);accum[i*10+c]=s[c];}
  });
  stage(net.trees.length+' cây · độ sâu 3',votes,[net.trees.length],'trees',{paths});stage('Phiếu tích lũy',accum,[net.trees.length,10],'votes');logits=stage('Điểm SAMME',s.map(v=>v/total),[10],'samme');probs=softmax(logits.map(v=>v/9));
 }else throw Error('Unknown model '+model);
 stage('Xác suất 0–9',probs,[10],'softmax',{input:logits,ada:model==='ada'});
 return{model,raw,x,stages,logits,probs,pred:probs.indexOf(Math.max(...probs))};
}
