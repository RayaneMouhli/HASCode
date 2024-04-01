import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import time
from scipy import interpolate
from SteeringPhases import SteeringPhasesPA8

mode = 'gpu'

if mode=='gpu':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # after switch device, you need restart the kernel
    torch.cuda.set_device(0)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_dtype(torch.float32)

erfa = scipy.io.loadmat("/home/sci/hdai/Projects/Datasets/has/Erfa.mat")
# mat = scipy.io.loadmat('/home/sci/hdai/Projects/Datasets/has//Modl.mat')
# model=torch.tensor(mat['Modl']-1,device=cuda)
mat = scipy.io.loadmat("/home/sci/hdai/Projects/Datasets/has/Modl_P6_half.mat")
tissue_mask = torch.tensor(mat["Modl"] - 1, device=device)

rho0 = 1e3
c0 = 1.5e3
fMHz = erfa["fMHz"][0][0]
# Pr = 100
Pr = 127.7  # for P6

offset_xmm = 0
offset_ymm = 0
# dmm = 50
dmm = 0  # for P6

h, v, z = 0, 0, 0

lmaxerfa, mmaxerfa, _ = erfa["perfa"].shape
Dyerfa = (erfa["Len"][0][0] / (lmaxerfa - 1)) * 1000
Dxerfa = (erfa["Len"][0][1] / (mmaxerfa - 1)) * 1000
yaxis_erfa = Dyerfa * torch.arange(-(lmaxerfa - 1) / 2, (lmaxerfa - 1) / 2 + 1)
xaxis_erfa = Dxerfa * torch.arange(-(mmaxerfa - 1) / 2, (mmaxerfa - 1) / 2 + 1)

xaxis_erfa_offs = xaxis_erfa + offset_xmm
yaxis_erfa_offs = yaxis_erfa + offset_ymm
lmax, mmax, nmax = tissue_mask.shape
# Dx, Dy, Dz = 0.5, 0.5, 0.5
Dx, Dy, Dz = 1, 1, 1
xaxisinterp = Dx * torch.arange(-(mmax - 1) / 2, (mmax - 1) / 2 + 1)
yaxisinterp = Dy * torch.arange(-(lmax - 1) / 2, (lmax - 1) / 2 + 1)
lx = ((mmax - 1) * Dx) / 1000
ly = ((lmax - 1) * Dy) / 1000
dz = Dz / 1000

emdist = dmm / 1000 - erfa["sx"][0, 0]

# Below is for P6
a_vect = torch.tensor([0, 0.043578, 0.0865, 0.071088, 0.21158], device=device)
aabs = torch.tensor([0, 0.070, 0.090, 0.071088, 0.21158], device=device)
c_vect = torch.tensor([1500, 1436, 1514, 1588.4, 1537], device=device)
randvc = torch.tensor([0, 0, 0, 0, 0], device=device)
rho_vect = torch.tensor([1e3, 0.928e3, 1.058e3, 1.090e3, 1.100e3], device=device)

a = torch.tensor(a_vect.cpu().numpy()[tissue_mask.cpu().numpy()], device="cuda")
rho = torch.tensor(rho_vect.cpu().numpy()[tissue_mask.cpu().numpy()], device="cuda")
c = torch.tensor(c_vect.cpu().numpy()[tissue_mask.cpu().numpy()], device="cuda")
a_list = [a[..., i] for i in range(a.shape[-1])]
rho_list = [rho[..., i] for i in range(rho.shape[-1])]
c_list = [c[..., i] for i in range(c.shape[-1])]

c = torch.ones_like(a, device="cuda")*1500
x_grid, y_grid, z_grid = torch.meshgrid(torch.linspace(-1, 1, c.shape[0]), torch.linspace(-1, 1, c.shape[1]), torch.linspace(-1, 1, c.shape[2]))
sigma = 0.2
gaussian = torch.exp(-(((x_grid) ** 2 + (y_grid) ** 2 + (z_grid) ** 2) / (2 * sigma ** 2))).to(device)
theta = torch.tensor([0.], device="cuda")
theta.requires_grad = True
for i in range(len(c_list)):
    c_list[i] = c[..., i]+ gaussian[..., i]*theta
    # c_list[i] = c[..., i]+ theta

# aabs = a
absmodl = aabs * 1e2 * fMHz
f = fMHz * 1e6

Z = torch.zeros(tissue_mask.shape, device=device).to(torch.complex128)
pfor = torch.zeros(tissue_mask.shape, device=device).to(torch.complex128)
pref = torch.zeros(tissue_mask.shape, device=device).to(torch.complex128)
Z_list = [torch.zeros(tissue_mask.shape[:2], device=device).to(torch.complex128) for _ in range(tissue_mask.shape[-1])]
pfor_list = [torch.zeros(tissue_mask.shape[:2], device=device).to(torch.complex128) for _ in range(tissue_mask.shape[-1])]
pref_list = [torch.zeros(tissue_mask.shape[:2], device=device).to(torch.complex128) for _ in range(tissue_mask.shape[-1])]
pfortot = torch.zeros(tissue_mask.shape, device=device).to(torch.complex128)
b_prime = torch.zeros(nmax, device=device).to(torch.complex128)
j = torch.complex(torch.Tensor([0]), torch.Tensor([1])).cuda()


ang = SteeringPhasesPA8(v, h, z, erfa["R"][0][0], erfa["ElemLoc"], fMHz * 1e6, c0)
angpg_vect = torch.exp(torch.complex(torch.zeros(ang.shape), ang))

start = time.time()

if erfa["isPA"][0][0]:
    angarr = torch.tile(angpg_vect, (lmaxerfa, mmaxerfa, 1)).to(device)
    serfa = torch.sum(torch.complex(torch.Tensor(erfa["perfa"].real), torch.Tensor(erfa["perfa"].imag)).to(device)*angarr, 2)
    serfa = serfa * torch.sqrt(torch.cuda.FloatTensor([Pr]))
else:
    serfa = torch.Tensor(erfa["perfa"]).to(device) * torch.sqrt(torch.cuda.FloatTensor([Pr]))


if torch.min(torch.abs(serfa)) == torch.zeros(1).to(device):
    f2 = interpolate.RegularGridInterpolator(
        (xaxis_erfa_offs, yaxis_erfa_offs),
        serfa.cpu().numpy(),
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
else:
    f2 = interpolate.RegularGridInterpolator(
        (xaxis_erfa_offs, yaxis_erfa_offs),
        serfa.cpu().numpy() / np.abs(serfa.cpu().numpy()),
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
amesh, bmesh = np.meshgrid(yaxisinterp, xaxisinterp, indexing='xy')
M = np.stack([amesh, bmesh], axis=-1)
zia = torch.tensor(f2(M).T).to(device)

za = torch.angle(zia)

f2m = interpolate.RegularGridInterpolator(
    (xaxis_erfa_offs, yaxis_erfa_offs),
    np.abs(serfa.cpu().numpy()),
    method="linear",
    bounds_error=False,
    fill_value=0,
)
ameshm, bmeshm = np.meshgrid(yaxisinterp, xaxisinterp, indexing='xy')
Mm = np.stack([ameshm, bmeshm], axis=-1)
zm = torch.tensor(f2m(Mm).T).to(device)

ppe = torch.conj(zm * torch.exp(za * j))

if emdist != 0:
    ferfa = torch.fft.fftshift(torch.fft.fft2(ppe))
    bprimeerfa = 2 * torch.pi * f / c0
    alpha = (
        torch.arange(1, mmax + 1, device="cuda")
        - torch.ceil(torch.Tensor([mmax / 2]).to(device))
    ) * (2 * np.pi / (bprimeerfa * lx))
    beta = (
        torch.arange(1, lmax + 1, device="cuda")
        - torch.ceil(torch.Tensor([lmax / 2]).to(device))
    ) * (2 * np.pi / (bprimeerfa * ly))
    alpha_sq, beta_sq = torch.meshgrid(alpha**2, beta**2, indexing="xy")
    expon = 1 - alpha_sq - beta_sq
    if emdist < 0:
        transferfa = torch.zeros(lmax, mmax, device=device).to(torch.complex64)
        transferfa[expon > 0] = torch.exp(
            j * bprimeerfa * emdist * torch.sqrt(expon[expon > 0])
        )
    else:
        transferfa = torch.exp(j * bprimeerfa * emdist * torch.sqrt(expon))
    pp = torch.fft.ifft2(torch.fft.ifftshift(ferfa * transferfa))
else:
    pp = ppe

A0 = torch.fft.fftshift(torch.fft.fft2(pp))
Z0 = rho0 * c0

criterion = torch.nn.MSELoss()

def forward_propagation(b_prime):
    for n in range(nmax):
        att_modl = torch.mul(a_list[n], 1e2 * fMHz)
        rho_modl = rho_list[n]
        # phase change: b_n(x,y)=2*\pi*f/c_n(x,y), Eq.(3)
        b = torch.mul(2 * torch.pi * f, torch.reciprocal(c_list[n]))
        Z_list[n] = torch.mul(
            torch.complex(torch.ones(b.shape, device="cuda"), -att_modl / b),
            torch.mul(rho_list[n], c_list[n]),
        )
        if n == 0:
            Refl = torch.div(torch.sub(Z_list[0], Z0), torch.add(Z_list[0], Z0))
            pforb = torch.mul(pp, torch.add(1, Refl))
        else:
            Refl = torch.div(
                torch.sub(Z_list[n], Z_list[n - 1]), torch.add(Z_list[n], Z_list[n - 1])
            )
            pref_list[n - 1] = torch.mul(Refl, pfor_list[n - 1])
            pforb = torch.mul(pfor_list[n - 1], torch.add(1, Refl))

        b_prime[n] = torch.sum(torch.mul(torch.abs(pforb), b)) / torch.sum(torch.abs(pforb))
        b_prime = b_prime.real
        alpha = torch.mul(
            torch.arange(1, mmax + 1, device="cuda")
            - torch.ceil(torch.Tensor([mmax / 2]).to(device)),
            torch.mul(2 * np.pi / lx, torch.reciprocal(b_prime[n])),
        )
        beta = torch.mul(
            torch.arange(1, lmax + 1, device="cuda")
            - torch.ceil(torch.Tensor([lmax / 2]).to(device)),
            torch.mul(2 * np.pi / ly, torch.reciprocal(b_prime[n])),
        )
        alpha = alpha.real
        beta = beta.real
        alpha_sq, beta_sq = torch.meshgrid(
            torch.pow(alpha, 2), torch.pow(beta, 2), indexing="xy"
        )

        expon = torch.sub(1, torch.add(alpha_sq, beta_sq))
        rp = dz * torch.sqrt(expon.to(torch.complex64))
        complex_idx = torch.imag(rp) > 0
        rp[complex_idx] = 0

        if n == 0 or torch.sum(torch.sum(torch.abs(A))):
            Aabs = torch.abs(A0)
        else:
            Aabs = torch.abs(A)
        Aabs[Aabs < 0.5 * torch.max(Aabs)] = 0
        Asum = torch.sum(Aabs)
        rpave = torch.div(torch.sum(torch.sum(rp * Aabs)), Asum)
        rpave = rpave.real
        b_vect = torch.mul(2 * torch.pi * f, torch.reciprocal(c_list[n]))
        a_vect = a_list[n] * 1e-4 * f
        a_vect = a_vect.real
        dbvect = torch.sub(b_vect, b_prime[n])
        dbvect = dbvect.real

        pprimeterm = torch.mul(
            torch.exp(torch.mul(dbvect, torch.mul(rpave, j))),
            torch.exp(torch.mul(-a_vect, rpave)),
        )

        # Eq. (7)
        p_prime = torch.mul(pforb, pprimeterm)
        # Eq. (9)
        A = torch.mul(
            torch.fft.fftshift(torch.fft.fft2(p_prime)),
            torch.exp(
                torch.mul(
                    b_prime[n] * dz, torch.mul(torch.sqrt(expon.to(torch.complex128)), j)
                )
            ),
        )

        pmat = torch.fft.ifft2(torch.fft.ifftshift(A))
        pfor_list[n] = pmat
        
    return pfor_list

pfor_list = forward_propagation(b_prime)

pfor_true = torch.load(f'Output/pfor_true.pt').to(device)
lamda_list = torch.zeros_like(c).to(torch.complex128)
lamda_list[..., -1] = 2*(pfor_list[-1]-pfor_true[..., -1])
pfor_true = torch.load(f'Output/pfor_true.pt').to(device)
# loss = criterion(pfor_list[-1], pfor_true[..., -1])
loss = ((pfor_list[-1]-pfor_true[..., -1]).real**2+(pfor_list[-1]-pfor_true[..., -1]).imag**2).sum()
loss.backward()

total_gradient = 0

for t in range(pfor.shape[-1]-1, 0, -1):
    f_t = pfor_list[t] - pfor_list[t-1]
    lamda = lamda_list[..., t]
    df_dtheta = torch.autograd.grad(f_t, theta, grad_outputs=lamda, allow_unused=True, retain_graph=True, create_graph=True)
    vjp_t = torch.autograd.grad(f_t, pfor_list[t], grad_outputs=lamda, allow_unused=True, retain_graph=True)
    lamda_list[..., t-1] = lamda_list[..., t]+vjp_t[0]
    f_t_prev = pfor_list[t-1] - pfor_list[t-2]
    total_gradient += lamda_list[..., t-1]*df_dtheta

print("The time difference is ", time.time() - start, "seconds")
torch.save(torch.stack(pfor_list, dim=-1).to(torch.complex128), f'Output/pout_pyth_P6__.pt')