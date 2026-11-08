import {preprocess} from './classical-train.js?v=20261104-theory';
export async function trainNeural({model:key,trainBytes,trainLabels,testBytes,testLabels,meta,count=2000,epochs=2,learningRate=.002,onUpdate=()=>{},shouldStop=()=>false,onStatus=()=>{}}){
 const tf=typeof window==='undefined'?await import('@tensorflow/tfjs'):await import('./vendor/tf.js');await tf.ready();onStatus('Backend '+tf.getBackend()+' · tải dữ liệu train');
 const model=tf.sequential();let layers;
 if(key==='cnn'){
  model.add(tf.layers.conv2d({inputShape:[28,28,1],filters:8,kernelSize:3,strides:1,padding:'same',activation:'relu',kernelInitializer:'heNormal'}));model.add(tf.layers.maxPooling2d({poolSize:2,strides:2}));model.add(tf.layers.conv2d({filters:16,kernelSize:3,strides:1,padding:'same',activation:'relu',kernelInitializer:'heNormal'}));model.add(tf.layers.maxPooling2d({poolSize:2,strides:2}));model.add(tf.layers.permute({dims:[3,1,2]}));model.add(tf.layers.flatten());model.add(tf.layers.dense({units:64,activation:'relu',kernelInitializer:'heNormal'}));model.add(tf.layers.dense({units:10,activation:'softmax'}));layers=[model.layers[0],model.layers[2],model.layers[6],model.layers[7]];
 }else if(key==='mlp'){
  model.add(tf.layers.dense({inputShape:[784],units:128,activation:'relu',kernelInitializer:'heNormal'}));model.add(tf.layers.dense({units:64,activation:'relu',kernelInitializer:'heNormal'}));model.add(tf.layers.dense({units:10,activation:'softmax'}));layers=model.layers;
 }else{model.add(tf.layers.dense({inputShape:[784],units:10,activation:'softmax',kernelInitializer:'zeros'}));layers=model.layers;}
 const optimizer=tf.train.adam(learningRate);model.compile({optimizer,loss:'categoricalCrossentropy',metrics:['accuracy']});
 const input=(bytes,n)=>key==='cnn'&&meta.models.cnn.preprocessing==='minmax'?Float32Array.from(bytes.subarray(0,n*784),v=>v/255):preprocess(bytes,meta,n);
 const tr=input(trainBytes,count),te=input(testBytes,1000),shape=key==='cnn'?[count,28,28,1]:[count,784],testShape=key==='cnn'?[1000,28,28,1]:[1000,784];
 const xs=tf.tensor(tr,shape),ys=tf.tidy(()=>tf.oneHot(tf.tensor1d(Int32Array.from(trainLabels.subarray(0,count)),'int32'),10)),tx=tf.tensor(te,testShape),ty=tf.tidy(()=>tf.oneHot(tf.tensor1d(Int32Array.from(testLabels.subarray(0,1000)),'int32'),10));
 function snapshot(){return tf.tidy(()=>{const w={};if(key==='lr'){const [W,b]=layers[0].getWeights(),wa=W.dataSync(),ba=b.dataSync(),out=new Float64Array(7850);out.set(wa);out.set(ba,7840);w['lr.W']=out;}
  else layers.forEach((l,idx)=>{const [W,b]=l.getWeights(),a=W.dataSync(),ba=b.dataSync();if(key==='cnn'&&idx<2){const [kh,kw,input,output]=W.shape,out=new Float64Array(a.length);for(let y=0;y<kh;y++)for(let x=0;x<kw;x++)for(let ic=0;ic<input;ic++)for(let oc=0;oc<output;oc++)out[((oc*input+ic)*kh+y)*kw+x]=a[((y*kw+x)*input+ic)*output+oc];w['cnn.c'+(idx+1)+'.weight']=out;w['cnn.c'+(idx+1)+'.bias']=Float64Array.from(ba);}
   else{const [input,output]=W.shape,out=new Float64Array(a.length);for(let i=0;i<input;i++)for(let o=0;o<output;o++)out[o*input+i]=a[i*output+o];const prefix=key==='mlp'?'mlp.'+[0,2,4][idx]:'cnn.f'+(idx-1);w[prefix+'.weight']=out;w[prefix+'.bias']=Float64Array.from(ba);}});return{w};});}
 const emit=data=>{const probe=tf.tidy(()=>Array.from(model.predict(tx.slice(testShape.map(()=>0),[1,...testShape.slice(1)])).dataSync()));return onUpdate({...data,probe});};
 let step=0,last=null,metrics=null;const batches=Math.ceil(count/32);try{
  await emit({patch:snapshot(),step:0,epoch:0,total:epochs*batches,loss:null,acc:null,metrics:null});
  for(let epoch=0;epoch<epochs;epoch++){
   const order=tf.util.createShuffledIndices(count);
   for(let batch=0;batch<batches;batch++){
    if(shouldStop())break;const idx=tf.tensor1d(Int32Array.from(order.subarray(batch*32,(batch+1)*32)),'int32'),bx=tf.gather(xs,idx),by=tf.gather(ys,idx);try{last=await model.trainOnBatch(bx,by);}finally{tf.dispose([idx,bx,by]);}step++;
    const update=step===1||step%Math.max(1,Math.floor(batches/6))===0||batch===batches-1||shouldStop();if(update){await emit({patch:snapshot(),step,epoch:epoch+1,total:epochs*batches,loss:Number(last[0]),acc:Number(last[1]),metrics:null});await tf.nextFrame();}
   }
   if(step){const ev=model.evaluate(tx,ty,{batchSize:100});try{metrics={loss:(await ev[0].data())[0],accuracy:(await ev[1].data())[0]};}finally{tf.dispose(ev);}await emit({patch:snapshot(),step,epoch:epoch+1,total:epochs*batches,loss:Number(last[0]),acc:Number(last[1]),metrics});}
   if(shouldStop())break;
  }
  // Export even when stopped between regular snapshots.
  const patch=snapshot();await emit({patch,step,epoch:null,total:epochs*batches,loss:last?Number(last[0]):null,acc:last?Number(last[1]):null,metrics});return{patch,step,metrics,stopped:shouldStop()};
 }finally{tf.dispose([xs,ys,tx,ty]);model.dispose();optimizer.dispose();}
}
