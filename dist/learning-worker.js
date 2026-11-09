import {TRACKS,BENCHMARK_TRACKS,createRoad} from './tracks.js';
import {createDemonstrations,splitDataset,demonstrationsFromRecordings} from './dataset.js';
import {MODEL_FORMAT,validateModel} from './network.js';
import {FEATURE_SCHEMA,FEATURE_NAMES} from './features.js';
import {evaluateModel} from './evaluate.js';
let busy=false;
const send=(type,data)=>self.postMessage({type,...data});
self.onmessage=async({data})=>{
 if(busy)return;busy=true;
 try{
  if(data.type==='evaluate'){const model=validateModel(data.model);send('progress',{stage:'Evaluating frozen weights on both tracks…',percent:10});const report=evaluateModel(model,BENCHMARK_TRACKS.map(createRoad));send('evaluated',{report});return;}
  if(data.type!=='train')throw new Error('Unknown learning operation.');
  send('progress',{stage:'Generating Track 1 expert examples and recovery states…',percent:2});
  const dataset=data.source==='manual'?demonstrationsFromRecordings(data.records||[],createRoad(TRACKS[0])):createDemonstrations(createRoad(TRACKS[0]),{count:6000,seed:2026,onProgress:p=>send('progress',{stage:'Collecting Track 1 demonstrations…',percent:Math.round(2+p*13)})});
  const split=splitDataset(dataset),base=data.fromScratch?null:validateModel(data.model),mean=base?.mean||split.mean,std=base?.std||split.std;
  send('progress',{stage:'Loading the training engine…',percent:16});
  await import('./vendor/tf.min.js');const tf=self.tf;await tf.setBackend('cpu');await tf.ready();
  const norm=xs=>xs.map(x=>x.map((v,i)=>(v-mean[i])/std[i]));
  const x=tf.tensor2d(norm(split.trainX)),y=tf.tensor2d(split.trainY),vx=tf.tensor2d(norm(split.valX)),vy=tf.tensor2d(split.valY);
  const model=tf.sequential();
  model.add(tf.layers.dense({inputShape:[8],units:base?.layers[0].outputs||48,activation:'tanh',kernelInitializer:tf.initializers.glorotUniform({seed:2026})}));
  model.add(tf.layers.dense({units:base?.layers[1].outputs||48,activation:'tanh',kernelInitializer:tf.initializers.glorotUniform({seed:2027})}));
  model.add(tf.layers.dense({units:2,activation:'linear',kernelInitializer:tf.initializers.glorotUniform({seed:2028})}));
  if(base){const weights=base.layers.flatMap(l=>[tf.tensor2d(l.weights,[l.inputs,l.outputs]),tf.tensor1d(l.biases)]);model.setWeights(weights);weights.forEach(t=>t.dispose());}
  const optimizer=tf.train.adam(base ? .00035 : .002);model.compile({optimizer,loss:'meanSquaredError'});
  const epochs=Math.max(5,Math.min(100,Number(data.epochs)||30)),history=[];let bestLoss=Infinity,bestWeights=null;
  await model.fit(x,y,{epochs,batchSize:256,shuffle:true,validationData:[vx,vy],callbacks:{onEpochEnd:async(epoch,logs)=>{
   const item={epoch:epoch+1,loss:logs.loss,valLoss:logs.val_loss};history.push(item);
   if(logs.val_loss<bestLoss){bestLoss=logs.val_loss;bestWeights?.forEach(w=>w.dispose());bestWeights=model.getWeights().map(w=>w.clone());}
   send('epoch',{item,percent:Math.round(18+(epoch+1)/epochs*65),stage:`Training · epoch ${epoch+1} / ${epochs}`,trainingSamples:split.trainX.length,validationSamples:split.valX.length});
   await new Promise(resolve=>setTimeout(resolve,0));
  }}});
  if(bestWeights)model.setWeights(bestWeights);
  const weights=model.getWeights(),layers=[];
  for(let i=0;i<weights.length;i+=2){const kernel=weights[i],bias=weights[i+1];layers.push({inputs:kernel.shape[0],outputs:kernel.shape[1],activation:i===4?'linear':'tanh',weights:Array.from(await kernel.data()),biases:Array.from(await bias.data())});}
  const trained=validateModel({format:MODEL_FORMAT,featureSchema:FEATURE_SCHEMA,featureNames:FEATURE_NAMES,simulatorVersion:'1.1.0',id:crypto.randomUUID(),createdAt:new Date().toISOString(),layers,mean,std,training:{tracks:dataset.tracks,seed:2026,samples:split.trainX.length,validationSamples:split.valX.length,epochs,optimizer:'Adam',loss:'mean squared error on steering and signed pedal',validationLoss:bestLoss,history,augmentation:dataset.augmentation,initialization:base?'continued from '+base.id:'random weights',parentModelId:base?.id||null},description:'State and mapped-road-geometry behavioral cloning. Trained only on Alpine Circuit. No reference-controller fallback.'});
  x.dispose();y.dispose();vx.dispose();vy.dispose();bestWeights?.forEach(w=>w.dispose());model.dispose();optimizer.dispose();
  send('progress',{stage:'Evaluating the trained model on both tracks…',percent:88});
  const report=evaluateModel(trained,BENCHMARK_TRACKS.map(createRoad));send('trained',{model:trained,report});
 }catch(error){send('error',{message:error.message||String(error)});}finally{busy=false;}
};
