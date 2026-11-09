import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createTraffic,stepTraffic,perception,lanePose,signalState,mod,ROAD,CROSSINGS,trafficNeural,trafficExpert,safetyBrake,TRAFFIC_FEATURES,validateTrafficModel} from '../dist/road-core.js';
import {evaluateTraffic} from '../dist/road-evaluate.js';
const model=validateTrafficModel(JSON.parse(fs.readFileSync(new URL('../dist/models/road-policy.json',import.meta.url),'utf8')));
test('Road has two lanes each way, continuous lane geometry, and pedestrian walk never conflicts with green',()=>{
 for(let lane=0;lane<4;lane++){assert.equal(lanePose(0,lane).direction,lane<2?1:-1);const a=lanePose(.001,lane),b=lanePose(ROAD.length-.001,lane);assert.ok(Math.hypot(a.x-b.x,a.z-b.z)<.01);}
 for(const cross of CROSSINGS)for(let time=0;time<96;time+=.1){const s=signalState(time,cross.offset);if(s.walk)assert.equal(s.phase,'red');}
});
test('each shared network forward pass responds to its own traffic observations',()=>{
 const w=createTraffic({count:3,supervisor:false}),a=w.cars[0],b=w.cars[1],lead=w.cars[2];a.s=b.s=50;a.lane=b.lane=0;a.direction=b.direction=1;a.speed=b.speed=10;b.lane=1;lead.lane=0;lead.direction=1;lead.s=mod(a.s+11);lead.speed=0;w.time=0;w.actors=[];
 const pa=perception(w,a),pb=perception(w,b);assert.ok(pa.gap<8&&pb.gap===80);const aa=trafficNeural(model,pa.features),ab=trafficNeural(model,pb.features);assert.ok(aa.pedal<ab.pedal-.2,'close follower must brake more than independent clear-lane car');
});
test('zeroed traffic weights produce zero controls without an expert or route fallback',()=>{
 const zero=structuredClone(model);zero.layers.forEach(l=>{l.weights.fill(0);l.biases.fill(0);});const w=createTraffic({count:1,supervisor:false});const car=w.cars[0];car.speed=0;
 for(let i=0;i<120;i++)stepTraffic(w,zero);assert.equal(w.stats.distance,0);assert.deepEqual(car.prediction,{steer:0,pedal:0});assert.equal(car.wheel,0);
});
test('crossings demand stopping, a stopped car holds its brakes, and supervisor changes remain separate',()=>{
 const w=createTraffic({count:1,supervisor:true}),car=w.cars[0],cross=CROSSINGS[0];car.s=mod(cross.s-30);car.speed=12;w.time=30;w.actors=[];const p=perception(w,car);assert.equal(p.stopRequired,true);assert.ok(trafficExpert(p.features).pedal<0);const command={steer:.17,pedal:1},applied=safetyBrake(car,p,command);assert.ok(applied.pedal<command.pedal);assert.equal(applied.steer,command.steer);assert.ok(applied.reason);assert.equal(command.pedal,1);
 car.s=mod(cross.s-8);car.speed=0;assert.ok(trafficExpert(perception(w,car).features).pedal<0);
});
test('contact detection records overlapping vehicles and vulnerable users even at rest',()=>{
 const w=createTraffic({count:2,supervisor:false}),a=w.cars[0],b=w.cars[1];a.lane=b.lane=0;a.direction=b.direction=1;a.s=b.s=100;a.speed=b.speed=0;w.paused=false;stepTraffic(w,model);assert.equal(w.stats.vehicleContacts,1);stepTraffic(w,model);assert.equal(w.stats.vehicleContacts,1,'persistent contact is one encounter');
 const p=createTraffic({count:1,supervisor:false}),car=p.cars[0];p.time=30;const actor=(awaitActors(p)).find(a=>a.inRoad);assert.ok(actor);let best=null;for(let lane=0;lane<4;lane++)for(const c of CROSSINGS){const pose=lanePose(c.s,lane),d=Math.hypot(pose.x-actor.x,pose.z-actor.z);if(!best||d<best.d)best={d,lane,s:c.s};}car.lane=best.lane;car.direction=car.lane<2?1:-1;car.s=best.s;car.speed=0;const pose=lanePose(car.s,car.lane);car.e=((actor.x-pose.x)*Math.cos(pose.yaw)-(actor.z-pose.z)*Math.sin(pose.yaw));stepTraffic(p,model);assert.ok(p.stats.vulnerableContacts>=1,'overlap with crossing actor must be counted');
});
function awaitActors(w){stepTraffic(w,model);return w.actors;}
test('traffic imports reject wrong input semantics, nonfinite weights, and invalid history',()=>{
 const wrong=structuredClone(model);wrong.featureNames[0]='camera-pixels';assert.throws(()=>validateTrafficModel(wrong));const bad=structuredClone(model);bad.layers[0].weights[0]=Infinity;assert.throws(()=>validateTrafficModel(bad));const hist=structuredClone(model);hist.training.history=[{loss:-1,valLoss:0}];assert.throws(()=>validateTrafficModel(hist));assert.equal(TRAFFIC_FEATURES.length,18);
});
test('frozen Road policy completes nine moving fleet scenarios with and without assistance',()=>{
 const before=JSON.stringify(model),report=evaluateTraffic(model);assert.equal(report.scenarios.length,9);for(const row of report.scenarios){assert.equal(row.passed,true,JSON.stringify(row));assert.ok(row.distanceKm>10);if(!row.supervisor)assert.equal(row.interventions,0);}assert.equal(JSON.stringify(model),before);assert.ok(report.scenarios.some(r=>r.dogScenario&&!r.supervisor));
});
