import * as THREE from './vendor/three.module.js';
export const TRACKS=[
 {id:'alpine-circuit',name:'Alpine Circuit',width:8,role:'train',seed:41,points:[[-65,-55],[-65,35],[-33,86],[40,92],[97,48],[95,-12],[53,-65],[9,-46],[-21,-79]]},
 {id:'canyon-run',name:'Canyon Run',width:7,role:'test',seed:83,points:[[-80,-60],[-94,8],[-57,64],[-17,44],[12,94],[64,68],[108,19],[67,-7],[81,-64],[31,-89],[1,-44],[-40,-86]]},
 {id:'harbor-sprint',name:'Harbor Sprint',width:7.5,role:'blind',biome:'coast',sealed:true,seed:542,points:[[-125,-85],[-125,-5],[-95,55],[-135,105],[-65,145],[5,110],[75,135],[145,95],[155,25],[105,10],[70,50],[35,30],[75,-30],[155,-65],[100,-130],[30,-120],[-5,-155],[-80,-135]]}
];
// The blind course is deliberately excluded from every automatic train/evaluate workflow.
export const BENCHMARK_TRACKS=TRACKS.filter(t=>!t.sealed);
export function createRoad(def){const curve=new THREE.CatmullRomCurve3(def.points.map(([x,z])=>new THREE.Vector3(x,0,z)),true,'centripetal');const points=curve.getSpacedPoints(900).map(p=>({x:p.x,z:p.z}));const distances=[0];for(let i=1;i<points.length;i++)distances.push(distances[i-1]+Math.hypot(points[i].x-points[i-1].x,points[i].z-points[i-1].z));return{...def,points,distances,length:distances.at(-1),curve};}
