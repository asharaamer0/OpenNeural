import test from 'node:test';
import assert from 'node:assert/strict';
import {BENCHMARK_TRACKS,createRoad} from '../dist/tracks.js';
import {FIXED_DT,initialState,step,nearest,pointAt,referenceControl,driveScore,manualSteering,steeringLimit} from '../dist/physics.js';
const roads=BENCHMARK_TRACKS.map(createRoad);
for(const road of roads)test(`reference completes ${road.name} without leaving the road`,()=>{const state=initialState(road);let maxDeviation=0;for(let i=0;i<120*180&&state.lap===1;i++){const input=referenceControl(state,road);assert.ok(Math.abs(input.steer)<=1&&input.throttle>=0&&input.throttle<=1&&input.brake>=0&&input.brake<=1);step(state,input,road);maxDeviation=Math.max(maxDeviation,Math.abs(state.deviation));}assert.equal(state.lap,2);assert.equal(state.collisions,0);assert.equal(state.offroadTime,0);assert.ok(maxDeviation<road.width/2-1);assert.ok(driveScore(state)>90);console.log(JSON.stringify({track:road.name,lapSeconds:state.finishedLaps[0].time,maxDeviation,score:driveScore(state)}));});
test('throttle accelerates and brake stops the vehicle',()=>{const r=roads[0],s=initialState(r);for(let i=0;i<120;i++)step(s,{steer:0,throttle:1,brake:0},r);assert.ok(s.speed>5);for(let i=0;i<120;i++)step(s,{steer:0,throttle:0,brake:1},r);assert.equal(s.speed,0);assert.ok(Number.isFinite(s.x)&&Number.isFinite(s.z));});
test('barriers contain an off-track vehicle and add a contact penalty',()=>{const r=roads[0],s=initialState(r),p=pointAt(r,30);s.x=p.x+Math.cos(p.heading)*20;s.z=p.z-Math.sin(p.heading)*20;step(s,{steer:0,throttle:0,brake:0},r);assert.ok(nearest(r,s.x,s.z).distance<=r.width/2+3.501);assert.equal(s.collisions,1);assert.ok(s.offroadTime>0);assert.ok(driveScore(s)<100);});
test('a car cannot complete a lap by crossing the start line without circuit travel',()=>{const r=roads[0],s=initialState(r);s.previousS=r.length-.1;step(s,{steer:0,throttle:0,brake:0},r);assert.equal(s.lap,1);});
test('fixed-step reference replay is deterministic',()=>{const r=roads[0],a=initialState(r),b=initialState(r);for(let i=0;i<600;i++){step(a,referenceControl(a,r),r,FIXED_DT);step(b,referenceControl(b,r),r,FIXED_DT);}assert.deepEqual(a,b);});
test('tracks are closed and physically distinct',()=>{assert.ok(roads[1].length>roads[0].length);for(const r of roads){assert.ok(Math.hypot(r.points[0].x-r.points.at(-1).x,r.points[0].z-r.points.at(-1).z)<.001);assert.ok(r.length>500);assert.equal(pointAt(r,0).x,pointAt(r,r.length).x);}});
const straight={points:[{x:0,z:-1000},{x:0,z:1000},{x:1000,z:1000},{x:1000,z:-1000},{x:0,z:-1000}],distances:[0,2000,3000,5000,6000],length:6000,width:40};
test('a 100 ms steering tap at 72 km/h produces a small correction in either direction',()=>{
 for(const direction of [-1,1]){
  const s=initialState(straight);s.z=0;s.speed=20;s.previousS=1000;
  let command=0;
  for(let i=0;i<84;i++){
   command=manualSteering(command,i<12?direction:0);
   step(s,{steer:command,throttle:0,brake:0},straight);
  }
  const headingDegrees=Math.abs(s.yaw*180/Math.PI);
  console.log(JSON.stringify({tapDirection:direction,headingChangeDegrees:headingDegrees,lateralMeters:Math.abs(s.x)}));
  assert.ok(headingDegrees<2,'brief taps must not cause sudden swerving');
  assert.ok(Math.abs(s.x)<.4);
  assert.equal(command,0);
  assert.equal(s.collisions,0);
 }
});
test('sustained steering remains responsive while high-speed lateral acceleration is bounded',()=>{
 const s=initialState(straight);s.z=0;s.speed=25;s.previousS=1000;
 let command=0,maxLateralAcceleration=0;
 for(let i=0;i<120;i++){
  command=manualSteering(command,1);
  step(s,{steer:command,throttle:0,brake:0},straight);
  maxLateralAcceleration=Math.max(maxLateralAcceleration,s.speed**2/2.7*Math.tan(Math.abs(s.steer)*.52));
 }
 assert.ok(s.yaw>.1,'holding a direction must still turn the vehicle');
 assert.ok(maxLateralAcceleration<=7.50001);
 assert.ok(steeringLimit(25)<steeringLimit(5));
});
