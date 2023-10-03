import torch

def SteeringPhasesPA8(v,h,z,R,ElemLoc,f,c0):
    thvect,phivect=ElemLoc[:,0],ElemLoc[:,1]
    numelem = thvect.shape[0]
    ang=torch.zeros(numelem)
    kk = 2*torch.pi*f/c0
    for g in range(numelem):
        tt = R * torch.sin(torch.cuda.FloatTensor([thvect[g]]))
        
        ss2 = R * torch.cos(torch.cuda.FloatTensor([thvect[g]]))
        ss3 = torch.sin(torch.cuda.FloatTensor([phivect[g]]))
        
        s = ss2*ss3
        
        aa3 = torch.cos(torch.cuda.FloatTensor([phivect[g]]))
        a = ss2 * aa3
        
        b = R - a
        d = R- (-z)
        
        r = torch.sqrt(torch.cuda.FloatTensor([(tt-v)**2 + (s+h)**2 + (d-b)**2]))
        
        ang[g]=kk*torch.round(torch.cuda.FloatTensor([r-R]))
    return ang