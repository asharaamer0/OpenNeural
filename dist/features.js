import {nearest,pointAt,wrapAngle} from './physics.js';
export const FEATURE_SCHEMA='road-relative-v1';
export const FEATURE_NAMES=['speed/30','wheel-angle/0.52','lane-offset/4','heading-error/0.5','lookahead-lateral/L','lookahead-forward/L','max-curvature/0.1','near-curvature/0.1'];
export function observe(state,road){
 const n=nearest(road,state.x,state.z),L=5+Math.abs(state.speed)*.72,p=pointAt(road,n.s+L),dx=p.x-state.x,dz=p.z-state.z;
 let maximum=0,near=0;
 for(let d=0;d<48;d+=6){const a=pointAt(road,n.s+d),b=pointAt(road,n.s+d+6),k=wrapAngle(b.heading-a.heading)/6;maximum=Math.max(maximum,Math.abs(k));if(d===0)near=k;}
 return [state.speed/30,state.steer,n.signed/4,wrapAngle(state.yaw-n.heading)/.5,(dx*Math.cos(state.yaw)-dz*Math.sin(state.yaw))/L,(dx*Math.sin(state.yaw)+dz*Math.cos(state.yaw))/L,maximum/.1,near/.1];
}
export function mirrorFeatures(x){return x.map((v,i)=>[1,2,3,4,7].includes(i)?-v:v);}
