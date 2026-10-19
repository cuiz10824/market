"""Small original PyTorch adaptations: five direct return heads, shared across indices.
Architecture references: LTSF-Linear, Bai et al. TCN, Nie et al. PatchTST.
These are project adaptations, not copied upstream packages or pretrained checkpoints.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LinearForecaster(nn.Module):
    def __init__(self,length,channels,kind,outputs):
        super().__init__();self.kind=kind
        self.linear=nn.Linear(length,outputs)
        self.trend=nn.Linear(length,outputs) if kind=='dlinear' else None
        self.mix=nn.Parameter(torch.zeros(channels))
        for layer in (self.linear,self.trend):
            if layer is not None:nn.init.normal_(layer.weight,std=.005);nn.init.zeros_(layer.bias)

    def forward(self,x):
        x=x.transpose(1,2)
        if self.kind=='nlinear':result=self.linear(x-x[:,:,-1:].detach())
        else:
            trend=F.avg_pool1d(F.pad(x,(3,3),mode='replicate'),7,stride=1)
            result=self.linear(x-trend)+self.trend(trend)
        return (result*self.mix.softmax(0)[None,:,None]).sum(1)


class CausalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.padding=2*dilation
        self.conv=nn.Conv1d(width,width,3,dilation=dilation)
        self.dropout=nn.Dropout(.1)

    def forward(self,x):
        return x+self.dropout(F.gelu(self.conv(F.pad(x,(self.padding,0)))))


class SmallTCN(nn.Module):
    def __init__(self,channels,outputs,width=8):
        super().__init__();self.input=nn.Conv1d(channels,width,1)
        self.blocks=nn.Sequential(*[CausalBlock(width,2**i) for i in range(8)])
        self.head=nn.Linear(width,outputs)
        nn.init.normal_(self.head.weight,std=.005);nn.init.zeros_(self.head.bias)

    def forward(self,x):return self.head(self.blocks(self.input(x.transpose(1,2)))[:,:,-1])


class SmallPatchTST(nn.Module):
    def __init__(self,length,channels,outputs,dimension=8,patch=8):
        super().__init__();self.patch=patch;self.extra=(-length)%patch
        count=(length+self.extra)//patch
        self.embedding=nn.Linear(patch,dimension)
        self.position=nn.Parameter(torch.zeros(1,count,dimension))
        nn.init.normal_(self.position,std=.01)
        layer=nn.TransformerEncoderLayer(dimension,2,dimension*2,dropout=.1,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False)
        self.head=nn.Linear(count*dimension,outputs)
        self.mix=nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.head.weight,std=.005);nn.init.zeros_(self.head.bias)

    def forward(self,x):
        x=x.transpose(1,2)
        if self.extra:x=F.pad(x,(self.extra,0),mode='replicate')
        patches=x.unfold(-1,self.patch,self.patch);b,c,n,_=patches.shape
        tokens=self.embedding(patches.reshape(b*c,n,self.patch))+self.position
        out=self.head(self.encoder(tokens).flatten(1)).reshape(b,c,-1)
        return (out*self.mix.softmax(0)[None,:,None]).sum(1)


class SharedForecaster(nn.Module):
    def __init__(self,architecture,length,channels,indices=8,outputs=5):
        super().__init__()
        if architecture in ('dlinear','nlinear'):self.network=LinearForecaster(length,channels,architecture,outputs)
        elif architecture=='tcn':self.network=SmallTCN(channels,outputs)
        elif architecture=='patchtst':self.network=SmallPatchTST(length,channels,outputs)
        else:raise ValueError('Unknown architecture')
        self.identity=nn.Embedding(indices,outputs);nn.init.zeros_(self.identity.weight)

    def forward(self,x,identity):return self.network(x)+self.identity(identity)
