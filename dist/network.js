import {observe,FEATURE_SCHEMA,FEATURE_NAMES} from './features.js';
import {clamp} from './physics.js';
export const MODEL_FORMAT='vector-behavioral-cloning-v1';
export function validateModel(model){
 if(model?.format!==MODEL_FORMAT||model.featureSchema!==FEATURE_SCHEMA||model.simulatorVersion!=='1.1.0')throw new Error('This model uses an incompatible observation or physics version.');
 if(!Array.isArray(model.layers)||model.layers.length!==3)throw new Error('Expected a three-layer neural network.');
 let input=FEATURE_NAMES.length;
 for(const [i,l] of model.layers.entries()){
  if(!Number.isInteger(l.outputs)||l.outputs<1||l.outputs>128||l.inputs!==input||!Array.isArray(l.weights)||l.weights.length!==l.inputs*l.outputs||!Array.isArray(l.biases)||l.biases.length!==l.outputs||![...l.weights,...l.biases].every(Number.isFinite)||l.activation!==(i===2?'linear':'tanh'))throw new Error('Invalid neural-network weights or architecture.');
  input=l.outputs;
 }
 if(input!==2||model.mean?.length!==8||model.std?.length!==8||!model.mean.every(Number.isFinite)||!model.std.every(v=>Number.isFinite(v)&&v>=.0001))throw new Error('Invalid model normalization.');
 if(!Array.isArray(model.training?.tracks)||model.training.tracks.length!==1||model.training.tracks[0]!=='alpine-circuit')throw new Error('This lab accepts models trained on Alpine Circuit only. Canyon Run must stay held out.');
 const history=model.training.history;
 if(typeof model.id!=='string'||model.id.length>200||!Number.isFinite(Date.parse(model.createdAt))||history!==undefined&&(!Array.isArray(history)||history.length>5000||history.some(h=>!Number.isFinite(h?.loss)||!Number.isFinite(h?.valLoss)||h.loss<0||h.valLoss<0)))throw new Error('Invalid model identity or training history.');
 return model;
}
// Pure forward inference. No expert controller, route replay, or reference action is used here.
export function predict(model,features){
 let a=features.map((x,i)=>(x-model.mean[i])/model.std[i]);
 for(const layer of model.layers){const b=new Array(layer.outputs);for(let j=0;j<layer.outputs;j++){let sum=layer.biases[j];for(let i=0;i<layer.inputs;i++)sum+=a[i]*layer.weights[i*layer.outputs+j];b[j]=layer.activation==='tanh'?Math.tanh(sum):sum;}a=b;}
 return a;
}
export function learnedControl(model,state,road){const [steering,pedal]=predict(model,observe(state,road));if(!Number.isFinite(steering)||!Number.isFinite(pedal))throw new Error('The model produced a non-finite action.');return {steer:clamp(steering,-1,1),throttle:clamp(pedal,0,1),brake:clamp(-pedal,0,1),handbrake:false};}
