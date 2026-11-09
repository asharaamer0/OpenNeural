import test from 'node:test';
import assert from 'node:assert/strict';
import {TRACKS,BENCHMARK_TRACKS,createRoad} from '../dist/tracks.js';
import {createDemonstrations} from '../dist/dataset.js';
import {evaluateModel} from '../dist/evaluate.js';
import {initialState} from '../dist/physics.js';
import {startTrial,finishTrial} from '../dist/blind-trial.js';
const definition=TRACKS.find(t=>t.sealed),road=createRoad(definition);
test('the unseen track is excluded from automatic training and evaluation',()=>{
 assert.equal(road.id,'harbor-sprint');assert.equal(road.role,'blind');assert.equal(road.biome,'coast');
 assert.equal(BENCHMARK_TRACKS.some(t=>t.id===road.id),false);
 assert.throws(()=>createDemonstrations(road),/held out/);
 // This must throw before any prediction, even without a valid model object.
 assert.throws(()=>evaluateModel(null,[road]),/sealed first-attempt/);
});
test('Harbor geometry is closed, finite, distinct, and does not intersect itself',()=>{
 assert.ok(road.length>900);assert.ok(road.width>=7);
 assert.deepEqual(road.points[0],road.points.at(-1));assert.ok(road.points.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.z)));
 const pts=road.points.filter((_,i)=>i%5===0),orient=(a,b,c)=>(b.x-a.x)*(c.z-a.z)-(b.z-a.z)*(c.x-a.x);
 for(let i=0;i<pts.length-1;i++)for(let j=i+3;j<pts.length-1;j++){
  if(i===0&&j>=pts.length-3)continue;
  const a=pts[i],b=pts[i+1],c=pts[j],d=pts[j+1];
  assert.ok(!(orient(a,b,c)*orient(a,b,d)<0&&orient(c,d,a)*orient(c,d,b)<0),'The course must not cross itself');
 }
 assert.notDeepEqual(definition.points,TRACKS[0].points);assert.notDeepEqual(definition.points,TRACKS[1].points);
});
test('first-run summaries preserve model identity and count interrupted attempts',()=>{
 const state=initialState(road),model={id:'frozen-model'},trial=startTrial(road,'learned',model,state,1);
 model.id='later-model';state.time=12;state.samples=12;state.errorSum=3;state.travel=road.length*.25;state.collisions=1;
 const result=finishTrial(trial,state,'interrupted: reset',road);
 assert.equal(result.modelId,'frozen-model');assert.equal(result.firstAttempt,true);assert.equal(result.completed,false);assert.equal(result.completionPercent,25);assert.equal(result.collisions,1);assert.equal(result.score,84);
});
