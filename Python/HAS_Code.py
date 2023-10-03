import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy
import scipy.io
import cmath
import time
from scipy import interpolate
from SteeringPhases import SteeringPhasesPA8

cuda=torch.device('cuda')

erfa = scipy.io.loadmat('../Setup/Erfa.mat')
# mat = scipy.io.loadmat('../Setup/Modl.mat')
# model=torch.tensor(mat['Modl']-1,device=cuda)
mat = scipy.io.loadmat('../Setup/Modl_P6.mat')
model=torch.tensor(mat['Modl']-1,device=cuda)


rho0 = 1e3
c0 = 1.5e3
fMHz = erfa['fMHz'][0][0]
# Pr = 100
Pr = 127.7 #for P6

offsetxmm = 0
offsetymm = 0
# dmm = 50
dmm = 0 #for P6

h,v,z =0,0,0

lmaxerfa, mmaxerfa,_=erfa['perfa'].shape
Dyerfa = (erfa['Len'][0][0]/(lmaxerfa-1)) * 1000
Dxerfa = (erfa['Len'][0][1]/(mmaxerfa-1)) * 1000
yaxiserfa = Dyerfa * torch.arange(-(lmaxerfa-1)/2,(lmaxerfa-1)/2+1)
xaxiserfa = Dxerfa * torch.arange(-(mmaxerfa-1)/2,(mmaxerfa-1)/2+1)

xaxiserfaoffs = xaxiserfa + offsetxmm
yaxiserfaoffs = yaxiserfa + offsetymm
lmax,mmax,nmax = model.shape
Dx, Dy, Dz = 0.5, 0.5, 0.5
xaxisinterp = Dx * torch.arange(-(mmax-1)/2,(mmax-1)/2+1)
yaxisinterp = Dy * torch.arange(-(lmax-1)/2,(lmax-1)/2+1)
lx = ((mmax - 1) * Dx)/1000
ly = ((lmax - 1) * Dy)/1000
dz = Dz/1000

emdist = dmm/1000 - erfa['sx'][0,0]


# avect = torch.tensor([0,0.133,0.091,0.086],device=cuda)
# aabs =  torch.tensor([0,0.133,0.091,0.086],device=cuda)
# cvect = torch.tensor([1500,1560,1480,1480],device=cuda)
# randvc = torch.tensor([0,0,0,0],device=cuda)
# rhovect = torch.tensor([1e3,1.064e3,0.937e3,0.937e3],device=cuda)

# Below is for P6
avect = torch.tensor([0,0.043578,0.0865,0.071088,0.21158],device=cuda)
aabs =  torch.tensor([0,0.070,0.090,0.071088,0.21158],device=cuda)
cvect = torch.tensor([1500,1436,1514,1588.4,1537],device=cuda)
randvc = torch.tensor([0,0,0,0,0],device=cuda)
rhovect = torch.tensor([1e3,0.928e3,1.058e3,1.090e3,1.100e3],device=cuda)

a = torch.tensor(avect.cpu().numpy()[model.cpu().numpy()],device='cuda')
rho = torch.tensor(rhovect.cpu().numpy()[model.cpu().numpy()],device='cuda')
c = torch.tensor(cvect.cpu().numpy()[model.cpu().numpy()],device='cuda')

# aabs = a
absmodl = aabs * 1e2 *fMHz
f = fMHz * 1e6

Z = torch.zeros(model.shape,device=cuda).to(torch.complex128)
pfor = torch.zeros(model.shape,device=cuda).to(torch.complex128)
pref =torch.zeros(model.shape,device=cuda).to(torch.complex128)
pfortot = torch.zeros(model.shape,device=cuda).to(torch.complex128)
bprime = torch.zeros(nmax,device=cuda).to(torch.complex128)
j=torch.complex(torch.Tensor([0]),torch.Tensor([1])).cuda()


ang = SteeringPhasesPA8(v,h,z,erfa['R'][0][0],erfa['ElemLoc'],fMHz*1e6,c0)
angpgvect=torch.exp(torch.complex(torch.zeros(ang.shape),ang))

start = time.time()

if erfa['isPA'][0][0]:
    angarr = torch.tile(angpgvect,(lmaxerfa,mmaxerfa,1)).to(cuda)
    serfa = torch.sum(torch.complex(torch.Tensor(erfa['perfa'].real),torch.Tensor(erfa['perfa'].imag)).to(cuda).to(cuda)*angarr,2)
    serfa = serfa * torch.sqrt(torch.cuda.FloatTensor([Pr]))
else :
    serfa = torch.Tensor(erfa['perfa']).to(cuda)*torch.sqrt(torch.cuda.FloatTensor([Pr]))


if torch.min(torch.abs(serfa))==torch.zeros(1).to(cuda):
    f2 = interpolate.RegularGridInterpolator((xaxiserfaoffs,yaxiserfaoffs),serfa.cpu().numpy(),method='linear',bounds_error=False,fill_value=0)
else :
    f2 = interpolate.RegularGridInterpolator((xaxiserfaoffs,yaxiserfaoffs),serfa.cpu().numpy()/np.abs(serfa.cpu().numpy()),method='linear', bounds_error=False,fill_value=0)
amesh,bmesh = np.meshgrid(yaxisinterp,xaxisinterp)
M = np.stack([amesh,bmesh],axis=-1)
zia = torch.tensor(f2(M).T).to(cuda)

za  = torch.angle(zia)

f2m = interpolate.RegularGridInterpolator((xaxiserfaoffs,yaxiserfaoffs),np.abs(serfa.cpu().numpy()),method='linear',bounds_error=False,fill_value=0)
ameshm,bmeshm = np.meshgrid(yaxisinterp,xaxisinterp)
Mm = np.stack([ameshm,bmeshm],axis=-1)
zm= torch.tensor(f2m(Mm).T).to(cuda)

ppe = torch.conj(zm * torch.exp(za*j))

if emdist !=0:
    ferfa = torch.fft.fftshift(torch.fft.fft2(ppe))
    bprimeerfa = 2 * torch.pi * f/c0
    alpha = (torch.arange(1,mmax+1,device='cuda') - torch.ceil(torch.Tensor([mmax/2]).to(cuda))) * ( 2 * np.pi/ (bprimeerfa*lx))
    beta = (torch.arange(1,lmax+1,device='cuda') - torch.ceil(torch.Tensor([lmax/2]).to(cuda))) * ( 2 * np.pi/ (bprimeerfa*ly))
    alpha_sq, beta_sq = torch.meshgrid(alpha**2,beta**2,indexing='xy')
    expon = 1 - alpha_sq - beta_sq
    if emdist<0:
        transferfa = torch.zeros(lmax,mmax,device=cuda).to(torch.complex64)
        transferfa[expon>0] = torch.exp(j * bprimeerfa * emdist * torch.sqrt(expon[expon>0]))
    else :
        transferfa = torch.exp(j * bprimeerfa * emdist * torch.sqrt(expon))
    pp = torch.fft.ifft2(torch.fft.ifftshift(ferfa*transferfa))
else :
    pp = ppe

pp=pp


A0 = torch.fft.fftshift(torch.fft.fft2(pp))
Z0 = rho0 * c0


for n in range(nmax):
    att_modl = torch.mul(a[:,:,n],1e2*fMHz)
    rho_modl = rho[:,:,n]
    b = torch.mul(2*math.pi *f,torch.reciprocal(c[:,:,n]))
    Z[:,:,n] = torch.mul(torch.complex(torch.ones(b.shape,device='cuda'), - att_modl/b) ,torch.mul(rho[:,:,n],c[:,:,n]))
    if n==0:
        
        Refl = torch.div(torch.sub(Z[:,:,0],Z0),torch.add(Z[:,:,0],Z0))
        pforb = torch.mul(pp,torch.add(1,Refl))
    else:

        Refl = torch.div(torch.sub(Z[:,:,n],Z[:,:,n-1]),torch.add(Z[:,:,n],Z[:,:,n-1]))
        pref[:,:,n-1] = torch.mul(Refl ,pfor[:,:,n-1])
        pforb = torch.mul(pfor[:,:,n-1],torch.add(1,Refl))

    bprime[n] = torch.sum(torch.mul(torch.abs(pforb),b))/torch.sum(torch.abs(pforb))
    bprime=bprime.real
    alpha = torch.mul( torch.arange(1,mmax+1,device='cuda') - torch.ceil(torch.Tensor([mmax/2]).to(cuda)) ,torch.mul(2 * np.pi/lx,torch.reciprocal(bprime[n])))
    beta = torch.mul(torch.arange(1,lmax+1,device='cuda') - torch.ceil(torch.Tensor([lmax/2]).to(cuda)),torch.mul(2 * np.pi/ly,torch.reciprocal(bprime[n])))
    alpha=alpha.real
    beta=beta.real
    alpha_sq, beta_sq = torch.meshgrid(torch.pow(alpha,2),torch.pow(beta,2),indexing='xy')

    expon = torch.sub(1,torch.add(alpha_sq,beta_sq))
    rp = dz * torch.sqrt(expon.to(torch.complex64))
    complex_idx = torch.imag(rp)>0
    rp[complex_idx] = 0

    if n==0 or torch.sum(torch.sum(torch.abs(A))):
        Aabs = torch.abs(A0)
    else :
        Aabs = torch.abs(A)
    Aabs[Aabs<0.5*torch.max(Aabs)]=0
    Asum = torch.sum(Aabs)
    rpave = torch.div(torch.sum(torch.sum(rp * Aabs)),Asum)
    rpave = rpave.real
    bvect = torch.mul(2 * math.pi * f,torch.reciprocal( c[:,:,n]))
    avect = a[:,:,n] * 1e-4 * f
    avect=avect.real
    dbvect = torch.sub(bvect ,bprime[n])
    dbvect=dbvect.real

    pprimeterm = torch.mul(torch.exp(torch.mul(dbvect,torch.mul(rpave,j))) ,torch.exp(torch.mul(-avect,rpave)))

    pprime = torch.mul(pforb, pprimeterm)
    A = torch.mul(torch.fft.fftshift(torch.fft.fft2(pprime)),torch.exp(torch.mul(bprime[n]*dz, torch.mul( torch.sqrt(expon.to(torch.complex128)),j))))

    pmat = torch.fft.ifft2(torch.fft.ifftshift(A))
    pfor[:,:,n] = pmat


print('The time difference is ',time.time()-start, 'seconds')

# scipy.io.savemat('../Output/pout_pyth.mat',{'pout_pyth':pfor.cpu().numpy()})
scipy.io.savemat('../Output/pout_pyth_P6.mat',{'pout_pyth':pfor.cpu().numpy()})
