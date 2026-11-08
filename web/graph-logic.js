// Exact dependencies used for Inspect; particles are separately labeled illustrative.
export function dependencies(result,stage,index){
 const stages=result.stages,s=stages[stage],links=[];if(!s)return links;
 const add=(source,i)=>{if(source>=0&&i>=0&&i<stages[source].data.length)links.push({stage:source,index:i});};
 if(s.op==='conv'){
  const side=s.shape[1],q=index%(side*side),y=Math.floor(q/side),x=q%side,[channels,ins]=s.inputShape,k=s.kernelSize||3,pad=s.padding||0;
  for(let c=0;c<channels;c++)for(let ky=0;ky<k;ky++)for(let kx=0;kx<k;kx++){const iy=y+ky-pad,ix=x+kx-pad;if(iy>=0&&iy<ins&&ix>=0&&ix<ins)add(stage-1,c*ins*ins+iy*ins+ix);}
 }else if(s.op==='linear'){
  Array.from(s.input,(v,i)=>[i,Math.abs(v*s.weights[index*s.input.length+i])]).sort((a,b)=>b[1]-a[1]).slice(0,8).forEach(([i])=>add(stage-1,i));
 }else if(s.op==='pool'){
  const side=s.shape[1],ins=s.inputShape[1],c=Math.floor(index/(side*side)),q=index%(side*side);[0,1,ins,ins+1].forEach(d=>add(stage-1,c*ins*ins+Math.floor(q/side)*2*ins+(q%side)*2+d));
 }else if(s.op==='softmax'){
  for(let i=0;i<stages[stage-1].data.length;i++)add(stage-1,i);
 }else if(s.op==='trees')s.paths[index].filter(n=>n.feature!==undefined).forEach(n=>add(stage-1,n.feature));
 else if(s.op==='bayes'||s.op==='lrterms')add(stage-1,index%784);
 else if(['flatten','relu','normalize'].includes(s.op))add(stage-1,index);
 return links;
}
// Valid extent of the 3×3 receptive field; outside-image padding remains zero.
export function convWindows(result,stage,index){const s=result.stages[stage];if(!s||s.op!=='conv'||s.inputShape[0]<2)return [];const [channels,side]=s.inputShape,q=index%(side*side),y=Math.floor(q/side),x=q%side,pad=s.padding||0,k=s.kernelSize||3;const top=Math.max(0,y-pad),left=Math.max(0,x-pad),bottom=Math.min(side-1,y-pad+k-1),right=Math.min(side-1,x-pad+k-1);return Array.from({length:channels},(_,channel)=>({stage:stage-1,channel,side,top,left,bottom,right}));}
