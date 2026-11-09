import json,time,uuid
from pathlib import Path
import numpy as np
import torch
from torch import nn
torch.set_num_threads(4);torch.manual_seed(6206);np.random.seed(6206)
d=json.loads(Path('training/road-data.json').read_text());mean=np.array(d['mean'],dtype=np.float32);std=np.array(d['std'],dtype=np.float32)
x=torch.tensor((np.array(d['trainX'],dtype=np.float32)-mean)/std);y=torch.tensor(d['trainY'],dtype=torch.float32);vx=torch.tensor((np.array(d['valX'],dtype=np.float32)-mean)/std);vy=torch.tensor(d['valY'],dtype=torch.float32)
net=nn.Sequential(nn.Linear(18,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,2));opt=torch.optim.Adam(net.parameters(),lr=.002);history=[];best=None;bestloss=1e9
for epoch in range(180):
    order=torch.randperm(len(x));total=0
    for ids in order.split(512):
        opt.zero_grad();p=net(x[ids]);loss=((p-y[ids])**2*torch.tensor([1.5,1.0])).mean();loss.backward();opt.step();total+=loss.item()*len(ids)
    with torch.no_grad():vl=((net(vx)-vy)**2).mean().item()
    history.append({'epoch':epoch+1,'loss':total/len(x),'valLoss':vl})
    if vl<bestloss:bestloss=vl;best={k:v.clone() for k,v in net.state_dict().items()}
    if epoch%20==0:print(json.dumps(history[-1]),flush=True)
    if epoch==139:
        for g in opt.param_groups:g['lr']=.0004
net.load_state_dict(best);layers=[]
for l in net:
    if isinstance(l,nn.Linear):layers.append({'inputs':l.in_features,'outputs':l.out_features,'activation':'linear' if l.out_features==2 else 'tanh','weights':l.weight.detach().numpy().T.flatten().tolist(),'biases':l.bias.detach().tolist()})
model={'format':'vector-road-policy-v1','simulatorVersion':'road-1.0.0','id':str(uuid.uuid4()),'createdAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'featureNames':d['metadata']['featureNames'],'mean':mean.tolist(),'std':std.tolist(),'layers':layers,'training':{**d['metadata'],'samples':len(x),'validationSamples':len(vx),'epochs':180,'seed':6206,'optimizer':'Adam','validationLoss':bestloss,'history':history}}
Path('dist/models/road-policy.json').write_text(json.dumps(model,separators=(',',':')));print(json.dumps({'saved':'road-policy.json','validationLoss':bestloss}),flush=True)
