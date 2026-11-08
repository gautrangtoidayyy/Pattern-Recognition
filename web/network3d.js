import * as THREE from './vendor/three.js';
import {dependencies,convWindows} from './graph-logic.js?v=20261104-bayes-restore';
const positive=new THREE.Color('#fb9148'),negative=new THREE.Color('#3887fa'),zero=new THREE.Color('#1c2940');
export class Network3D {
 constructor(canvas,onSelect){
  this.canvas=canvas;canvas.dataset.renderer='webgl';this.onSelect=onSelect;this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#0a101c');this.camera=new THREE.PerspectiveCamera(42,1,.1,4000);
  this.renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false});this.renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));this.renderer.setClearColor('#0a101c');
  this.controls=new THREE.OrbitControls(this.camera,canvas);this.controls.enableDamping=true;this.controls.dampingFactor=.09;this.controls.minDistance=8;this.controls.maxDistance=1800;
  this.scene.add(new THREE.AmbientLight(0xffffff,1.7));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(100,180,160);this.scene.add(light);
  const grid=new THREE.GridHelper(900,90,0x20314b,0x142137);grid.position.y=-43;this.scene.add(grid);
  this.group=new THREE.Group();this.scene.add(this.group);this.flow=new THREE.Group();this.scene.add(this.flow);this.flowStart=0;this.overlays=new THREE.Group();this.scene.add(this.overlays);this.geometry=new THREE.BoxGeometry(1,1,1);this.meshes=[];this.layers=[];this.dummy=new THREE.Object3D();this.raycaster=new THREE.Raycaster();this.tween=null;this.result=null;
  this.marker=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshBasicMaterial({color:0xffdc80,wireframe:true,depthTest:false}));this.marker.visible=false;this.marker.renderOrder=5;this.scene.add(this.marker);
  let down;canvas.addEventListener('pointerdown',e=>down=[e.clientX,e.clientY]);canvas.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;const r=canvas.getBoundingClientRect();this.raycaster.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),this.camera);const hit=this.raycaster.intersectObjects(this.meshes,false)[0];if(hit?.instanceId!==undefined)this.onSelect(hit.object.userData.stage,hit.instanceId);});
  new ResizeObserver(()=>this.resize()).observe(canvas.parentElement);this.resize();this.animate();
 }
 resize(){const r=this.canvas.parentElement.getBoundingClientRect();this.renderer.setSize(Math.max(1,r.width),Math.max(1,r.height),false);this.camera.aspect=Math.max(1,r.width)/Math.max(1,r.height);this.camera.updateProjectionMatrix();if(this.result)this.renderer.render(this.scene,this.camera);}
 label(text,digit=false){const c=document.createElement('canvas');c.width=digit?64:512;c.height=digit?64:100;const ctx=c.getContext('2d');ctx.fillStyle='#b1c4df';ctx.font=digit?'bold 44px system-ui':'500 23px system-ui';ctx.textAlign='center';ctx.fillText(text,c.width/2,digit?49:48);const tex=new THREE.CanvasTexture(c),sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false}));sprite.scale.set(digit?2.2:32,digit?2.2:6.25,1);return sprite;}

 setNetwork(result){
  const structure=result.model+':'+result.stages.map(s=>s.data.length).join(',');const rebuild=this.structure!==structure;this.result=result;
  if(rebuild){this.clearGroup(this.group);this.meshes=[];this.layers=[];this.structure=structure;let cursor=0;
   result.stages.forEach((s,stage)=>{
    let channels=1,rows,cols,mapCols=1,mapRows=1,cell;
    if(s.shape.length===3){[channels,rows,cols]=s.shape;mapCols=channels>1?Math.min(4,channels):1;mapRows=Math.ceil(channels/mapCols);cell=channels>1?.30:.86;}
    else if(s.op==='flatten'){cols=s.data.length;rows=1;cell=78.4/cols;}else{cols=s.op==='votes'?10:s.data.length===10?1:Math.ceil(Math.sqrt(s.data.length));rows=Math.ceil(s.data.length/cols);cell=s.data.length===10?1.7:s.data.length>2000?.46:.90;}
    const gap=channels>1?2.2:0,width=(cols*cell+gap)*mapCols-gap,height=(rows*cell+gap)*mapRows-gap;
    const L={x:cursor,channels,rows,cols,cell,mapCols,mapRows,width,height,gap,stage,strip:s.op==='flatten'};this.layers.push(L);
    const mat=new THREE.MeshStandardMaterial({roughness:.75,metalness:.12});const mesh=new THREE.InstancedMesh(this.geometry,mat,s.data.length);mesh.userData.stage=stage;mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);mesh.frustumCulled=false;this.group.add(mesh);this.meshes.push(mesh);
    const label=this.label(`${String(stage+1).padStart(2,'0')} · ${s.name}`);label.position.set(cursor,-height/2-7,0);this.group.add(label);const sub=this.label(s.shape.join(' × '));sub.scale.multiplyScalar(.7);sub.position.set(cursor,-height/2-11,0);this.group.add(sub);
    if(stage){const prev=this.layers[stage-1],from=new THREE.Vector3(prev.x+3,0,0),to=new THREE.Vector3(cursor-3,0,0);const a=new THREE.ArrowHelper(to.clone().sub(from).normalize(),from,to.distanceTo(from),0x4f6384,1.8,1);this.group.add(a);}
    if(stage===result.stages.length-1&&s.data.length===10){for(let digit=0;digit<10;digit++){const tag=this.label(String(digit),true);tag.position.copy(this.position(stage,digit)).add(new THREE.Vector3(1,0,2.8));this.group.add(tag);}}
    cursor+=Math.max(27,Math.min(46,width*.3+27));
   });this.total=cursor-27;
  }
  result.stages.forEach((s,stage)=>{
   const mesh=this.meshes[stage],L=this.layers[stage];let max=0;for(const v of s.data)max=Math.max(max,Math.abs(v));L.max=max||1;
   for(let i=0;i<s.data.length;i++){const p=this.position(stage,i),v=s.data[i],t=Math.min(1,Math.abs(v)/L.max);this.dummy.position.copy(p);this.dummy.scale.set(.13+t*(L.strip?.9:L.cell*2.5),L.strip?1.4:L.cell*.82,L.cell*.9);this.dummy.updateMatrix();mesh.setMatrixAt(i,this.dummy.matrix);const c=zero.clone().lerp(v<0?negative:positive,Math.pow(t,.65));if(s.op==='raw')c.setRGB(.08+t*.9,.10+t*.9,.15+t*.85);mesh.setColorAt(i,c);}
   mesh.instanceMatrix.needsUpdate=true;mesh.instanceColor.needsUpdate=true;
  });
  if(rebuild){this.overview();this.camera.position.copy(this.tween.to);this.controls.target.copy(this.tween.targetTo);this.tween=null;this.controls.update();}this.renderer.render(this.scene,this.camera);
 }
 animateStep(from,to){
  this.clearGroup(this.flow);this.flowStart=0;
  if(!this.result||from===to||(this.result.model==='cnn'&&to>=3&&to<=7)||matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  if(this.result.model!=='cnn'&&this.result.stages[to].op==='flatten'&&to>0)this.flattenMorph={start:performance.now(),source:to-1,target:to};
  const a=from,b=to,sa=this.result.stages[a],sb=this.result.stages[b];
  // Representative samples communicate progression; displayed data remain the actual inference values.
  const count=12;this.flowPaths=[];
  for(let i=0;i<count;i++){const ai=Math.min(sa.data.length-1,Math.floor((i+.5)*sa.data.length/count)),bi=Math.min(sb.data.length-1,Math.floor((i+.5)*sb.data.length/count));const start=this.position(a,ai),end=this.position(b,bi);const dot=new THREE.Mesh(new THREE.SphereGeometry(.65,8,6),new THREE.MeshBasicMaterial({color:sb.data[bi]<0?0x58a5ff:0xffb45e,transparent:true,opacity:.9,depthTest:false}));dot.renderOrder=8;this.flow.add(dot);this.flowPaths.push({start,end,dot});}
  this.flowStart=performance.now();
 }
 stopFlow(){this.clearGroup(this.flow);this.flowStart=0;this.finishFlattenMorph();}
 finishFlattenMorph(){if(!this.flattenMorph)return;this.updateFlattenMorph(1);this.flattenMorph=null;}
 updateFlattenMorph(t){const {source,target}=this.flattenMorph,L=this.layers[target],data=this.result.stages[target].data,mesh=this.meshes[target],u=1-(1-t)**3;for(let i=0;i<data.length;i++){const p=this.position(source,i).lerp(this.position(target,i),u),v=data[i],k=Math.min(1,Math.abs(v)/L.max);this.dummy.position.copy(p);this.dummy.scale.set(.13+k*.9,1.4,L.cell*.9);this.dummy.updateMatrix();mesh.setMatrixAt(i,this.dummy.matrix);}mesh.instanceMatrix.needsUpdate=true;}
 position(stage,index){const L=this.layers[stage],plane=L.rows*L.cols,c=Math.floor(index/plane),q=index%plane;const col=q%L.cols,row=Math.floor(q/L.cols),tileX=c%L.mapCols,tileY=Math.floor(c/L.mapCols);return new THREE.Vector3(L.x,L.height/2-(tileY*(L.rows*L.cell+L.gap)+row*L.cell+L.cell/2),-L.width/2+tileX*(L.cols*L.cell+L.gap)+col*L.cell+L.cell/2);}
 select(stage,index){if(!this.result)return;this.clearGroup(this.overlays);const L=this.layers[stage],p=this.position(stage,index);this.marker.position.copy(p);this.marker.scale.set(L.strip?1.3:L.cell*3.1,L.strip?1.8:L.cell*1.3,L.strip?.25:L.cell*1.3);this.marker.visible=true;const s=this.result.stages[stage];
  const points=[];for(const link of dependencies(this.result,stage,index))points.push(this.position(link.stage,link.index),p.clone());
  for(const w of convWindows(this.result,stage,index)){
   const cell=this.layers[w.stage].cell,a=this.position(w.stage,w.channel*w.side*w.side+w.top*w.side+w.left),b=this.position(w.stage,w.channel*w.side*w.side+w.bottom*w.side+w.right),x=a.x+.5;
   const yTop=a.y+cell/2,yBottom=b.y-cell/2,zLeft=a.z-cell/2,zRight=b.z+cell/2;
   const outline=[new THREE.Vector3(x,yTop,zLeft),new THREE.Vector3(x,yTop,zRight),new THREE.Vector3(x,yBottom,zRight),new THREE.Vector3(x,yBottom,zLeft),new THREE.Vector3(x,yTop,zLeft)];
   const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(outline),new THREE.LineBasicMaterial({color:0xffe283,depthTest:false}));line.renderOrder=7;this.overlays.add(line);
  }
  if(points.length)this.overlays.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0xffa154,transparent:true,opacity:.65})));
 }
 focus(stage,channel=null){if(!this.layers[stage])return;const L=this.layers[stage];let target=new THREE.Vector3(L.x,0,0),size=Math.max(20,L.height,L.width);if(channel!==null&&L.channels>1){target=this.position(stage,channel*L.rows*L.cols+Math.floor(L.rows/2)*L.cols+Math.floor(L.cols/2));size=Math.max(10,L.rows*L.cell*1.7);}
  const distance=size*(this.camera.aspect<1?2.6:1.8);this.fly(target,target.clone().add(new THREE.Vector3(distance*.9,distance*.35,distance*.65)));
 }
 focusPair(a,ai,b,bi){const p=this.position(a,ai),q=this.position(b,bi),target=p.clone().add(q).multiplyScalar(.5),size=Math.max(35,p.distanceTo(q)+25),distance=size*(this.camera.aspect<1?2.5:1.7);this.fly(target,target.clone().add(new THREE.Vector3(distance*.8,distance*.3,distance*.8)));}
 overview(){const center=new THREE.Vector3(this.total/2,0,0),distance=Math.max(110,this.total/(Math.min(2,this.camera.aspect)*.55));this.fly(center,center.clone().add(new THREE.Vector3(distance*.4,distance*.32,distance*.85)));}
 fly(target,pos){this.tween={start:performance.now(),from:this.camera.position.clone(),to:pos,targetFrom:this.controls.target.clone(),targetTo:target};}
 clearGroup(group){for(const c of [...group.children]){group.remove(c);c.traverse(o=>{if(o.geometry&&o.geometry!==this.geometry)o.geometry.dispose();if(o.material){const a=Array.isArray(o.material)?o.material:[o.material];a.forEach(m=>{m.map?.dispose();m.dispose();});}});}}
 animate(){requestAnimationFrame(()=>this.animate());if(this.flattenMorph){const t=Math.min(1,(performance.now()-this.flattenMorph.start)/1500);this.updateFlattenMorph(t);if(t===1)this.flattenMorph=null;}if(this.flowStart){const elapsed=performance.now()-this.flowStart;for(let i=0;i<this.flowPaths.length;i++){const {start,end,dot}=this.flowPaths[i],t=Math.max(0,Math.min(1,(elapsed-i*55)/1050));dot.position.lerpVectors(start,end,t);dot.visible=elapsed>=i*55&&t<1;}if(elapsed>1800)this.stopFlow();}if(this.tween){const t=Math.min(1,(performance.now()-this.tween.start)/650),u=1-(1-t)**3;this.camera.position.lerpVectors(this.tween.from,this.tween.to,u);this.controls.target.lerpVectors(this.tween.targetFrom,this.tween.targetTo,u);if(t===1)this.tween=null;}this.controls.update();this.renderer.render(this.scene,this.camera);}
}
