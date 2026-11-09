import {initialState,pointAt,referenceControl,step,FIXED_DT,steeringLimit} from './physics.js';
import {observe,mirrorFeatures} from './features.js';
export function seededRandom(seed){return()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296;};}
export function createDemonstrations(road,{count=12000,seed=2026,onProgress=()=>{}}={}){
 if(road.role!=='train'||road.id!=='alpine-circuit')throw new Error('Only Alpine Circuit can generate training examples. Canyon Run is held out.');
 const random=seededRandom(seed),x=[],y=[],groups=[];
 const add=(state,group)=>{const f=observe(state,road),a=referenceControl(state,road),label=[a.steer,a.throttle-a.brake];x.push(f);y.push(label);groups.push(group);x.push(mirrorFeatures(f));y.push([-label[0],label[1]]);groups.push(group);};
 // First collect a clean, complete expert lap under the simulator's dynamics.
 const expert=initialState(road);let ticks=0;
 while(expert.lap===1&&ticks<120*180){const a=referenceControl(expert,road);if(ticks%6===0)add(expert,'expert-lap');step(expert,a,road);ticks++;}
 if(expert.lap!==2||expert.collisions||expert.offroadTime)throw new Error('Reference demonstration did not complete a clean lap.');
 // Teacher-labelled perturbations teach recovery, speed changes, and both turn directions.
 const cleanSamples=x.length;
 for(let i=0;i<count;i++){
  const s=random()*road.length,p=pointAt(road,s),state=initialState(road),offset=(random()-.5)*4.8,angle=(random()-.5)*.65;
  state.x=p.x+Math.cos(p.heading)*offset;state.z=p.z-Math.sin(p.heading)*offset;state.yaw=p.heading+angle;state.speed=random()*30;state.steer=(random()-.5)*2*steeringLimit(state.speed)/.52;
  add(state,`perturbation-${i}`);
  if(i%500===0)onProgress(i/count);
 }
 return {x,y,groups,seed,tracks:[road.id],cleanSamples,perturbationSamples:count*2,reference:{time:expert.finishedLaps[0].time,score:expert.finishedLaps[0].score,offroadSeconds:expert.offroadTime,collisions:expert.collisions},augmentation:'teacher-labelled position/heading/speed perturbations and mirrored steering'};
}
export function splitDataset(data,{seed=714}={}){
 // Split grouped pairs together so a mirrored example cannot leak into validation.
 const rand=seededRandom(seed),groupSplit=new Map(),trainX=[],trainY=[],valX=[],valY=[];
 data.x.forEach((f,i)=>{const g=data.groups[i];if(!groupSplit.has(g))groupSplit.set(g,rand()<.15);if(groupSplit.get(g)){valX.push(f);valY.push(data.y[i]);}else{trainX.push(f);trainY.push(data.y[i]);}});
 if(!valX.length||!trainX.length)throw new Error('The training dataset is too small.');
 const mean=Array(8).fill(0),std=Array(8).fill(0);for(const x of trainX)for(let i=0;i<8;i++)mean[i]+=x[i]/trainX.length;for(const x of trainX)for(let i=0;i<8;i++)std[i]+=(x[i]-mean[i])**2/trainX.length;for(let i=0;i<8;i++)std[i]=Math.max(.01,Math.sqrt(std[i]));
 return {trainX,trainY,valX,valY,mean,std};
}
export function demonstrationsFromRecordings(records,road){
 const x=[],y=[],groups=[];let accepted=0;
 for(const r of records){
  if(r.driver!=='manual'||r.track?.id!==road.id||r.simulatorVersion!=='1.1.0')continue;
  let count=0;
  for(const [i,f] of r.frames.entries()){
   if(f.action?.handbrake)continue;
   const a=f.action,o=f.observation;
   if(!a||!o||![a.steering,a.throttle,a.brake,o.x,o.z,o.yaw,o.speedMps].every(Number.isFinite))continue;
   const features=Array.isArray(f.features)&&f.features.length===8?f.features:observe({x:o.x,z:o.z,yaw:o.yaw,speed:o.speedMps,steer:(o.wheelSteeringRad||0)/.52},road);
   if(!features.every(Number.isFinite)||[a.steering,a.throttle,a.brake].some(v=>Math.abs(v)>1))continue;
   const group=`${r.id}-chunk-${Math.floor(i/100)}`,label=[a.steering,a.throttle-a.brake];
   x.push(features);y.push(label);groups.push(group);x.push(mirrorFeatures(features));y.push([-label[0],label[1]]);groups.push(group);count++;
  }
  if(count)accepted++;
 }
 if(x.length<600)throw new Error('Record at least 15 seconds of manual driving on Track 1 with the current physics before training. Older physics, Track 2, neural drives, and handbrake frames are excluded.');
 return {x,y,groups,tracks:[road.id],seed:2026,recordings:accepted,augmentation:'Mirrored manual examples; contiguous frame chunks kept together during validation split'};
}
