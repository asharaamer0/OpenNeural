import {initialState,pointAt,step,referenceControl,FIXED_DT,driveScore} from './physics.js';
import {learnedControl} from './network.js';
export function runAttempt(road,model,{offset=0,heading=0,speed=0,startS=0,maxSeconds=180,reference=false}={}){
 const state=initialState(road),p=pointAt(road,startS);state.x=p.x+Math.cos(p.heading)*offset;state.z=p.z-Math.sin(p.heading)*offset;state.yaw=p.heading+heading;state.speed=speed;state.previousS=startS;
 const trace=[];let action,actionError=0,actionSamples=0,peakDeviation=0;
 for(let i=0;i<maxSeconds*120;i++){
  if(i%6===0){action=reference?referenceControl(state,road):learnedControl(model,state,road);const teacher=referenceControl(state,road);actionError+=(Math.abs(action.steer-teacher.steer)+Math.abs((action.throttle-action.brake)-(teacher.throttle-teacher.brake)))/2;actionSamples++;if(i%24===0)trace.push({s:state.previousS,deviation:state.deviation,speed:state.speed,t:state.time});}
  step(state,action,road,FIXED_DT);peakDeviation=Math.max(peakDeviation,Math.abs(state.deviation));
  if(state.travel>=road.length*.995&&startS>0||state.lap>1&&startS===0)break;
  if(state.collisions>=8||state.offroadTime>15||state.time>10&&state.speed<.15)break;
 }
 const completed=startS>0?state.travel>=road.length*.995:state.lap>1;
 return {trackId:road.id,track:road.name,trackLengthM:road.length,completed,completionPercent:completed?100:Math.min(99.9,Math.max(0,state.travel/road.length*100)),lapSeconds:state.time,score:driveScore(state),meanDeviationM:state.errorSum/Math.max(state.samples,.001),maxDeviationM:peakDeviation,offroadSeconds:state.offroadTime,collisions:state.collisions,actionMAE:actionError/Math.max(actionSamples,1),offset,heading,speed,startS,trace};
}
export function compareAttempt(attempt,baseline){
 let total=0;for(const p of attempt.trace){let best=baseline.trace[0],distance=Infinity;for(const b of baseline.trace){const raw=Math.abs(p.s-b.s),diff=Math.min(raw,baseline.trackLengthM-raw);if(diff<distance){best=b;distance=diff;}}total+=Math.abs(p.deviation-best.deviation);}
 return {...attempt,baselineLapSeconds:baseline.lapSeconds,lapTimeDifference:attempt.lapSeconds-baseline.lapSeconds,trajectoryMAEM:total/Math.max(attempt.trace.length,1)};
}
export function evaluateModel(model,roads){if(roads.some(r=>r.sealed))throw new Error('Harbor Sprint is a sealed first-attempt track. Automatic evaluation is disabled; test it live.');const rows=[];for(const road of roads){const baseline=runAttempt(road,null,{reference:true});const nominal=compareAttempt(runAttempt(road,model),baseline);const perturbed=[{offset:.75,heading:.07,speed:6},{offset:-.75,heading:-.07,speed:6},{offset:1,heading:.1,speed:8,startS:road.length*.25},{offset:-1,heading:-.1,speed:8,startS:road.length*.6}].map(options=>compareAttempt(runAttempt(road,model,options),baseline));rows.push({trackId:road.id,track:road.name,role:road.role,baseline,nominal,perturbed,passed:[nominal,...perturbed].filter(r=>r.completed&&r.collisions===0&&r.offroadSeconds===0).length,total:5});}return {modelId:model.id,createdAt:new Date().toISOString(),method:'Frozen weights; 20 Hz neural controls; 120 Hz physics; one nominal and four perturbed starts per track; no reference fallback.',trainingTracks:model.training.tracks,tracks:rows};}
