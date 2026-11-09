import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {BENCHMARK_TRACKS,createRoad} from '../dist/tracks.js';
import {initialState,pointAt} from '../dist/physics.js';
import {observe} from '../dist/features.js';
import {validateModel,learnedControl} from '../dist/network.js';
import {createDemonstrations,splitDataset,demonstrationsFromRecordings} from '../dist/dataset.js';
import {evaluateModel,runAttempt} from '../dist/evaluate.js';
const model=validateModel(JSON.parse(fs.readFileSync(new URL('../dist/models/baseline.json',import.meta.url),'utf8'))),roads=BENCHMARK_TRACKS.map(createRoad);
test('the frozen learned network completes both tracks and recovers from all specified starts',()=>{
 const before=JSON.stringify(model),report=evaluateModel(model,roads);
 assert.deepEqual(report.trainingTracks,['alpine-circuit']);
 for(const track of report.tracks){assert.equal(track.passed,5);assert.equal(track.total,5);for(const attempt of [track.nominal,...track.perturbed]){assert.equal(attempt.completed,true);assert.equal(attempt.collisions,0);assert.equal(attempt.offroadSeconds,0);assert.ok(attempt.score>=90);assert.ok(attempt.trajectoryMAEM<.15);}}
 assert.equal(JSON.stringify(model),before,'evaluation must never update learned weights');
});
test('removing learned weights stops the driver instead of silently falling back to the expert',()=>{
 const zero=structuredClone(model);zero.layers.forEach(l=>{l.weights.fill(0);l.biases.fill(0);});
 const attempt=runAttempt(roads[1],zero);
 assert.equal(attempt.completed,false);assert.equal(attempt.completionPercent,0);
});
test('road-relative observations and neural actions are invariant to translation and rotation',()=>{
 const road=roads[0],state=initialState(road),p=pointAt(road,70);state.x=p.x;state.z=p.z;state.yaw=p.heading+.08;state.speed=12;
 const angle=1.4,cos=Math.cos(angle),sin=Math.sin(angle),transform=p=>({x:p.x*cos+p.z*sin+250,z:-p.x*sin+p.z*cos-380});
 const movedRoad={...road,points:road.points.map(transform)},movedState={...state,...transform(state),yaw:state.yaw+angle};
 const a=observe(state,road),b=observe(movedState,movedRoad);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-9));
 const aa=learnedControl(model,state,road),bb=learnedControl(model,movedState,movedRoad);assert.ok(Math.abs(aa.steer-bb.steer)<1e-9);assert.ok(Math.abs(aa.throttle-bb.throttle)<1e-9);
});
test('training data rejects the held-out track and keeps mirrored pairs together',()=>{
 assert.throws(()=>createDemonstrations(roads[1],{count:20}),/held out/);
 const data=createDemonstrations(roads[0],{count:40});assert.deepEqual(data.tracks,['alpine-circuit']);
 const split=splitDataset(data);assert.ok(split.trainX.length&&split.valX.length);assert.equal(split.trainX.length%2,0);assert.equal(split.valX.length%2,0);
 assert.ok(split.mean.every(Number.isFinite)&&split.std.every(v=>Number.isFinite(v)&&v>0));
});
test('manual ingestion rejects test-track, old-physics, and neural recordings',()=>{
 const p=pointAt(roads[0],35),frame={observation:{x:p.x,z:p.z,yaw:p.heading,speedMps:8,wheelSteeringRad:0},action:{steering:.1,throttle:.2,brake:0,handbrake:false}};
 const base={id:'manual-1',driver:'manual',simulatorVersion:'1.1.0',track:{id:'alpine-circuit'},frames:Array.from({length:500},()=>structuredClone(frame))};
 const data=demonstrationsFromRecordings([base,{...base,track:{id:'canyon-run'}},{...base,simulatorVersion:'1.0.0'},{...base,driver:'learned'}],roads[0]);
 assert.equal(data.recordings,1);assert.equal(data.x.length,1000);assert.deepEqual(data.tracks,['alpine-circuit']);
 assert.throws(()=>demonstrationsFromRecordings([{...base,driver:'learned'}],roads[0]),/Record at least/);
});
test('model imports validate weights, feature schema, and training-track metadata',()=>{
 const invalid=structuredClone(model);invalid.layers[0].weights[0]=Infinity;assert.throws(()=>validateModel(invalid),/Invalid neural/);
 const leaked=structuredClone(model);leaked.training.tracks.push('canyon-run');assert.throws(()=>validateModel(leaked),/held out/);
 const wrong=structuredClone(model);wrong.featureSchema='vision';assert.throws(()=>validateModel(wrong),/incompatible/);
 const corruptHistory=structuredClone(model);corruptHistory.training.history='invalid';assert.throws(()=>validateModel(corruptHistory),/training history/);
});
