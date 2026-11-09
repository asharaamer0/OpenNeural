import {roadDataset} from './road-data.js';
import {validateTrafficModel} from './road-core.js';
import {evaluateTraffic} from './road-evaluate.js';
const send=(type,data)=>self.postMessage({type,...data});
let busy=false;
self.onmessage=async({data})=>{
 if(busy)return;busy=true;
 try{
  const base=validateTrafficModel(data.model);
  const evaluate=model=>evaluateTraffic(model,{onProgress:({completed,total})=>send('progress',{percent:86+completed/total*14,stage:`Fleet evaluation ${completed} / ${total} · frozen weights`})});
  if(data.type==='evaluate'){send('evaluated',{report:evaluate(base)});return;}
  if(data.type!=='train')throw new Error('Unknown Road learning operation.');
  send('progress',{percent:2,stage:'Collecting expert traffic demonstrations and queue rollouts…'});
  const d=roadDataset(6000,20261017,40),mean=base.mean,std=base.std;
  send('progress',{percent:10,stage:`Collected ${d.metadata.examples.toLocaleString()} Road examples. Loading TensorFlow.js…`});
  await import('./vendor/tf.min.js');const tf=self.tf;await tf.setBackend('cpu');await tf.ready();
  const normalize=xs=>xs.map(row=>row.map((v,i)=>(v-mean[i])/std[i]));
  const x=tf.tensor2d(normalize(d.trainX)),y=tf.tensor2d(d.trainY),vx=tf.tensor2d(normalize(d.valX)),vy=tf.tensor2d(d.valY),net=tf.sequential();
  base.layers.forEach((layer,i)=>net.add(tf.layers.dense({...(i===0?{inputShape:[18]}:{}),units:layer.outputs,activation:layer.activation})));
  const initial=base.layers.flatMap(l=>[tf.tensor2d(l.weights,[l.inputs,l.outputs]),tf.tensor1d(l.biases)]);net.setWeights(initial);initial.forEach(t=>t.dispose());
  const optimizer=tf.train.adam(.0002);net.compile({optimizer,loss:'meanSquaredError'});
  const initialLoss=net.evaluate(vx,vy);let bestLoss=(await initialLoss.data())[0],best=net.getWeights().map(t=>t.clone()),bestEpoch=0;initialLoss.dispose();const history=[];
  await net.fit(x,y,{epochs:20,batchSize:256,shuffle:true,validationData:[vx,vy],callbacks:{onEpochEnd:async(epoch,logs)=>{
   const item={epoch:epoch+1,loss:logs.loss,valLoss:logs.val_loss};history.push(item);
   if(logs.val_loss<bestLoss){bestLoss=logs.val_loss;bestEpoch=epoch+1;best.forEach(t=>t.dispose());best=net.getWeights().map(t=>t.clone());}
   send('epoch',{item,percent:12+(epoch+1)/20*72,stage:`Road training · epoch ${epoch+1} / 20`});await new Promise(r=>setTimeout(r,0));
  }}});
  net.setWeights(best);const weights=net.getWeights(),layers=[];
  for(let i=0;i<weights.length;i+=2)layers.push({inputs:weights[i].shape[0],outputs:weights[i].shape[1],activation:i===4?'linear':'tanh',weights:Array.from(await weights[i].data()),biases:Array.from(await weights[i+1].data())});
  const model=validateTrafficModel({...base,id:crypto.randomUUID(),createdAt:new Date().toISOString(),layers,training:{...d.metadata,samples:d.trainX.length,validationSamples:d.valX.length,epochs:20,optimizer:'TensorFlow.js / Adam',validationLoss:bestLoss,bestEpoch,history,parentModelId:base.id,initialization:'Continued from current weights; best validation checkpoint including initial weights retained.'}});
  x.dispose();y.dispose();vx.dispose();vy.dispose();best.forEach(t=>t.dispose());net.dispose();optimizer.dispose();
  send('trained',{model,report:evaluate(model)});
 }catch(error){send('error',{message:error.message||String(error)});}finally{busy=false;}
};
