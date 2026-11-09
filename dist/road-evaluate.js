import {createTraffic,stepTraffic,requestDog,trafficSummary,validateTrafficModel,TRAFFIC_VERSION} from './road-core.js';
export function evaluateTraffic(model,{seconds=180,onProgress=()=>{}}={}){
 validateTrafficModel(model);
 const cases=[{count:12,seed:42},{count:16,seed:73},{count:24,seed:91},{count:16,seed:731,dog:true}];
 const configs=[...cases.map(c=>({...c,supervisor:true})),...cases.map(c=>({...c,supervisor:false})),{count:16,seed:73,supervisor:false,mode:'expert'}],rows=[];
 for(const [index,config] of configs.entries()){
  const world=createTraffic(config);let minGap=80,maxLaneError=0;
  for(let tick=0;tick<seconds*60;tick++){
   if(config.dog&&tick===30*60)requestDog(world);stepTraffic(world,model);
   for(const car of world.cars){minGap=Math.min(minGap,car.observation?.gap??80);maxLaneError=Math.max(maxLaneError,Math.abs(car.e));}
  }
  const s=trafficSummary(world),row={...s,dogScenario:!!config.dog,minGap,maxLaneError,passed:s.vehicleContacts===0&&s.vulnerableContacts===0&&s.redLights===0&&s.laneSeconds===0&&s.speedSeconds===0&&s.unsafeSeconds===0&&s.distance>500};
  rows.push(row);onProgress({completed:index+1,total:configs.length,row});
 }
 return {simulatorVersion:TRAFFIC_VERSION,modelId:model.id,createdAt:new Date().toISOString(),secondsPerScenario:seconds,notes:'Deterministic simulation cases. A pass requires zero contacts, red-light crossings, lane/speed/headway violation time, and over 500 fleet meters. Supervisor interventions are counted separately. Success is limited to these cases, not universal safety or morality.',scenarios:rows};
}
