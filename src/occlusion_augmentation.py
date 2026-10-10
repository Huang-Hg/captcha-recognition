"""Position-preserving occlusion views for supervised CTC consistency training."""
import math
import torch


def occlude(images, valid_widths, generator, probability=.65):
    batch,channels,height,width=images.shape
    device=images.device
    def rand(*shape):
        return torch.rand(*shape,device=device,generator=generator)
    x=torch.arange(width,device=device)[None,None,:]
    y=torch.arange(height,device=device)[None,:,None]
    active=rand(batch,1,1)<probability
    kind=torch.randint(3,(batch,1,1),device=device,generator=generator)
    phase=rand(batch,1,1)*2*math.pi
    base=height*(.3+.4*rand(batch,1,1))
    amplitude=height*(.04+.1*rand(batch,1,1))
    freq=(.7+rand(batch,1,1)*1.5)*2*math.pi/valid_widths[:,None,None]
    center=base+amplitude*torch.sin(x*freq+phase)
    thickness=.65+rand(batch,1,1)*1.35
    wave=(y-center).abs()<thickness
    # Erasures and diagonal interruptions occupy at most a few rows per column.
    cuts=((y-base).abs()<.7)|((y-(base+height*.19)).abs()<.7)
    diagonal=(y-(base+(x-valid_widths[:,None,None]/2)*.1)).abs()<1.0
    mask=torch.where(kind==0,cuts,torch.where(kind==1,wave,diagonal))
    mask=mask&active&(x<valid_widths[:,None,None])
    color=rand(batch,channels,1,1)*1.6-.8
    color=torch.where((kind==0)[:,None],torch.ones_like(color),color)
    return torch.where(mask[:,None],color,images)
