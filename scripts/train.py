"""Supervised behavioral cloning on Alpine Circuit only. Exports real trained weights."""
import json, time, uuid, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
torch.set_num_threads(4)
torch.manual_seed(2026)
np.random.seed(2026)
data=json.loads(Path('training/dataset.json').read_text())
mean=np.array(data['mean'],dtype=np.float32);std=np.array(data['std'],dtype=np.float32)
x=torch.tensor((np.array(data['trainX'],dtype=np.float32)-mean)/std);y=torch.tensor(data['trainY'],dtype=torch.float32)
vx=torch.tensor((np.array(data['valX'],dtype=np.float32)-mean)/std);vy=torch.tensor(data['valY'],dtype=torch.float32)
net=nn.Sequential(nn.Linear(8,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,2))
optimizer=torch.optim.Adam(net.parameters(),lr=.002)
history=[];best_loss=float('inf');best=None
for epoch in range(100):
    order=torch.randperm(len(x));total=0
    for ids in order.split(512):
        optimizer.zero_grad();pred=net(x[ids]);loss=((pred-y[ids])**2).mean();loss.backward();optimizer.step();total+=loss.item()*len(ids)
    with torch.no_grad():
        vloss=((net(vx)-vy)**2).mean().item();mae=(net(vx)-vy).abs().mean().item()
    history.append({'epoch':epoch+1,'loss':total/len(x),'valLoss':vloss,'valMAE':mae})
    if vloss<best_loss:best_loss=vloss;best={k:v.clone() for k,v in net.state_dict().items()}
    if epoch%10==0:print(json.dumps(history[-1]),flush=True)
    if epoch==59:
        for g in optimizer.param_groups:g['lr']=.0005
net.load_state_dict(best)
layers=[]
for layer in net:
    if isinstance(layer,nn.Linear):layers.append({'inputs':layer.in_features,'outputs':layer.out_features,'activation':'linear' if layer.out_features==2 else 'tanh','weights':layer.weight.detach().numpy().T.flatten().tolist(),'biases':layer.bias.detach().tolist()})
model={'format':'vector-behavioral-cloning-v1','featureSchema':'road-relative-v1','simulatorVersion':'1.1.0','id':str(uuid.uuid4()),'createdAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'layers':layers,'mean':mean.tolist(),'std':std.tolist(),'training':{**data['metadata'],'samples':len(x),'validationSamples':len(vx),'epochs':len(history),'optimizer':'Adam','loss':'mean squared error on steering and signed pedal','seed':2026,'validationLoss':best_loss,'history':history},'description':'A state and mapped-road-geometry behavioral-cloning network, trained on Alpine Circuit only; no reference-controller fallback.'}
Path('dist/models').mkdir(exist_ok=True)
Path('dist/models/baseline.json').write_text(json.dumps(model,separators=(',',':')))
print(json.dumps({'saved':'dist/models/baseline.json','validationLoss':best_loss,'parameters':sum(p.numel() for p in net.parameters())}),flush=True)
