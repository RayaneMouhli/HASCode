import matplotlib.pyplot as plt
from jax import numpy as jnp
import jax
import numpy as np
import scipy.io
import time
from scipy import interpolate
from steering_phases_jax import steering_phases_PA8

erfa = scipy.io.loadmat("/home/sci/hdai/Projects/Datasets/has/Erfa.mat")
# mat = scipy.io.loadmat('/home/sci/hdai/Projects/Datasets/has//Modl.mat')
# model=jnp.array(mat['Modl']-1,device=cuda)
mat = scipy.io.loadmat("/home/sci/hdai/Projects/Datasets/has/Modl_P6.mat")
tissue_mask = jnp.array(mat["Modl"] - 1)

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
yaxis_erfa = Dyerfa * jnp.arange(-(lmaxerfa - 1) / 2, (lmaxerfa - 1) / 2 + 1)
xaxis_erfa = Dxerfa * jnp.arange(-(mmaxerfa - 1) / 2, (mmaxerfa - 1) / 2 + 1)

xaxis_erfa_offs = xaxis_erfa + offset_xmm
yaxis_erfa_offs = yaxis_erfa + offset_ymm
lmax, mmax, nmax = tissue_mask.shape
Dx, Dy, Dz = 0.5, 0.5, 0.5
xaxisinterp = Dx * jnp.arange(-(mmax - 1) / 2, (mmax - 1) / 2 + 1)
yaxisinterp = Dy * jnp.arange(-(lmax - 1) / 2, (lmax - 1) / 2 + 1)
lx = ((mmax - 1) * Dx) / 1000
ly = ((lmax - 1) * Dy) / 1000
dz = Dz / 1000

emdist = dmm / 1000 - erfa["sx"][0, 0]

# Below is for P6
a_vect = jnp.array([0, 0.043578, 0.0865, 0.071088, 0.21158], dtype=jnp.float64)
aabs = jnp.array([0, 0.070, 0.090, 0.071088, 0.21158], dtype=jnp.float64)
c_vect = jnp.array([1500, 1436, 1514, 1588.4, 1537], dtype=jnp.float64)
randvc = jnp.array([0, 0, 0, 0, 0], dtype=jnp.float64)
rho_vect = jnp.array([1e3, 0.928e3, 1.058e3, 1.090e3, 1.100e3], dtype=jnp.float64)

a = jnp.array(a_vect[tissue_mask], dtype=jnp.float64)
rho = jnp.array(rho_vect[tissue_mask], dtype=jnp.float64)
c = jnp.array(c_vect[tissue_mask], dtype=jnp.float64)

# aabs = a
absmodl = aabs * 1e2 * fMHz
f = fMHz * 1e6

Z = jnp.zeros(tissue_mask.shape, dtype=jnp.complex128)
pfor = jnp.zeros(tissue_mask.shape, dtype=jnp.complex128)
pref = jnp.zeros(tissue_mask.shape, dtype=jnp.complex128)
pfortot = jnp.zeros(tissue_mask.shape, dtype=jnp.complex128)
b_prime = jnp.zeros(nmax, dtype=jnp.complex128)
# j = jnp.vectorize(complex)(jnp.array([0]), jnp.array([1]))
j = 1j

ang = steering_phases_PA8(v, h, z, erfa["R"][0][0], erfa["ElemLoc"], fMHz * 1e6, c0)
angpg_vect = jnp.exp(jnp.zeros(ang.shape)+1j*ang)

start = time.time()

if erfa["isPA"][0][0]:
    angarr = jnp.tile(angpg_vect, (lmaxerfa, mmaxerfa, 1))
    serfa = jnp.sum((jnp.array(erfa["perfa"].real)+1j*jnp.array(erfa["perfa"].imag))*angarr, 2)
    serfa = serfa * jnp.sqrt(jnp.array([Pr]))
else:
    serfa = jnp.array(erfa["perfa"]) * jnp.sqrt(jnp.array([Pr]))


if jnp.min(jnp.abs(serfa)) == jnp.zeros(1):
    f2 = interpolate.RegularGridInterpolator(
        (np.asarray(xaxis_erfa_offs), np.asarray(yaxis_erfa_offs)),
        np.asarray(serfa),
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
else:
    f2 = interpolate.RegularGridInterpolator(
        (np.asarray(xaxis_erfa_offs), np.asarray(yaxis_erfa_offs)),
        np.asarray((serfa / jnp.abs(serfa))),
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
amesh, bmesh = jnp.meshgrid(yaxisinterp, xaxisinterp, indexing='xy')
M = jnp.stack([amesh, bmesh], axis=-1)
zia = jnp.array(f2(np.asarray(M)).T)

za = jnp.angle(zia)

f2m = interpolate.RegularGridInterpolator(
    (np.asarray(xaxis_erfa_offs), np.asarray(yaxis_erfa_offs)),
    np.asarray(jnp.abs(serfa)),
    method="linear",
    bounds_error=False,
    fill_value=0,
)
ameshm, bmeshm = jnp.meshgrid(yaxisinterp, xaxisinterp, indexing='xy')
Mm = jnp.stack([ameshm, bmeshm], axis=-1)
zm = jnp.array(f2m(np.asarray(Mm)).T)

ppe = jnp.conj(zm * jnp.exp(za * j))

if emdist != 0:
    ferfa = jnp.fft.fftshift(jnp.fft.fft2(ppe))
    bprimeerfa = 2 * jnp.pi * f / c0
    alpha = (
        jnp.arange(1, mmax + 1)
        - jnp.ceil(jnp.array([mmax / 2]))
    ) * (2 * jnp.pi / (bprimeerfa * lx))
    beta = (
        jnp.arange(1, lmax + 1)
        - jnp.ceil(jnp.array([lmax / 2]))
    ) * (2 * jnp.pi / (bprimeerfa * ly))
    alpha_sq, beta_sq = jnp.meshgrid(alpha**2, beta**2, indexing="xy")
    expon = 1 - alpha_sq - beta_sq
    if emdist < 0:
        transferfa = jnp.zeros((lmax, mmax))# 
        # transferfa[expon > 0] = jnp.exp(j * bprimeerfa * emdist * jnp.sqrt(expon[expon > 0]))
        transferfa = transferfa.at[expon > 0].set(jnp.exp(j * bprimeerfa * emdist * jnp.sqrt(expon[expon > 0])))
    else:
        transferfa = jnp.exp(j * bprimeerfa * emdist * jnp.sqrt(expon))
    pp = jnp.fft.ifft2(jnp.fft.ifftshift(ferfa * transferfa))
else:
    pp = ppe

A0 = jnp.fft.fftshift(jnp.fft.fft2(pp))
Z0 = rho0 * c0

def forward_propagation(c, b_prime, Z, pfor, pref):
    for n in range(nmax):
        att_modl = jnp.multiply(a[:, :, n], 1e2 * fMHz)
        rho_modl = rho[:, :, n]
        # phase change: b_n(x,y)=2*\pi*f/c_n(x,y), Eq.(3)
        b = jnp.multiply(2 * jnp.pi * f, jnp.reciprocal(c[:, :, n]))
        # Z[:, :, n] = jnp.multiply(jnp.vectorize(complex)(jnp.ones(b.shape), -att_modl / b), jnp.multiply(rho[:, :, n], c[:, :, n]))
        Z = Z.at[:,:,n].set(jnp.multiply(jnp.exp(jnp.ones(b.shape)-1j*att_modl / b), jnp.multiply(rho[:, :, n], c[:, :, n])+1j*0))
        if n == 0:
            Refl = jnp.divide(jnp.subtract(Z[:, :, 0], Z0), jnp.add(Z[:, :, 0], Z0))
            pforb = jnp.multiply(pp, jnp.add(1, Refl))
        else:
            Refl = jnp.divide(
                jnp.subtract(Z[:, :, n], Z[:, :, n - 1]), jnp.add(Z[:, :, n], Z[:, :, n - 1])
            )
            # pref[:, :, n - 1] = jnp.multiply(Refl, pfor[:, :, n - 1])
            pref = pref.at[:,:,n-1].set(jnp.multiply(Refl, pfor[:, :, n - 1]))
            pforb = jnp.multiply(pfor[:, :, n - 1], jnp.add(1, Refl))

        # b_prime[n] = jnp.sum(jnp.multiply(jnp.abs(pforb), b)) / jnp.sum(jnp.abs(pforb))
        b_prime = b_prime.at[n].set(jnp.sum(jnp.multiply(jnp.abs(pforb), b)) / jnp.sum(jnp.abs(pforb))).real
        # b_prime = b_prime.real
        alpha = jnp.multiply(
            jnp.arange(1, mmax + 1)
            - jnp.ceil(jnp.array([mmax / 2])),
            jnp.multiply(2 * jnp.pi / lx, jnp.reciprocal(b_prime[n])),
        )
        beta = jnp.multiply(
            jnp.arange(1, lmax + 1)
            - jnp.ceil(jnp.array([lmax / 2])),
            jnp.multiply(2 * jnp.pi / ly, jnp.reciprocal(b_prime[n])),
        )
        alpha = alpha.real
        beta = beta.real
        alpha_sq, beta_sq = jnp.meshgrid(
            jnp.power(alpha, 2), jnp.power(beta, 2), indexing="xy"
        )

        expon = jnp.subtract(1, jnp.add(alpha_sq, beta_sq))
        rp = dz * jnp.sqrt(expon.astype(jnp.complex128))
        complex_idx = jnp.imag(rp) > 0
        # rp[complex_idx] = 0
        rp = rp.at[complex_idx].set(0)

        if n == 0 or jnp.sum(jnp.sum(jnp.abs(A))):
            Aabs = jnp.abs(A0)
        else:
            Aabs = jnp.abs(A)
        # Aabs[Aabs < 0.5 * jnp.max(Aabs)] = 0
        Aabs = Aabs.at[Aabs < 0.5 * jnp.max(Aabs)].set(0)

        Asum = jnp.sum(Aabs)
        rpave = jnp.divide(jnp.sum(jnp.sum(rp * Aabs)), Asum)
        rpave = rpave.real
        b_vect = jnp.multiply(2 * jnp.pi * f, jnp.reciprocal(c[:, :, n]))
        a_vect = a[:, :, n] * 1e-4 * f
        a_vect = a_vect.real
        dbvect = jnp.subtract(b_vect, b_prime[n])
        dbvect = dbvect.real

        pprimeterm = jnp.multiply(
            jnp.exp(jnp.multiply(dbvect, jnp.multiply(rpave, j))),
            jnp.exp(jnp.multiply(-a_vect, rpave)),
        )

        # Eq. (7)
        p_prime = jnp.multiply(pforb, pprimeterm)
        # Eq. (9)
        A = jnp.multiply(
            jnp.fft.fftshift(jnp.fft.fft2(p_prime)),
            jnp.exp(
                jnp.multiply(
                    b_prime[n] * dz, jnp.multiply(jnp.sqrt(expon.astype(jnp.complex128)), j)
                )
            ),
        )

        pmat = jnp.fft.ifft2(jnp.fft.ifftshift(A))
        # pfor[:, :, n] = pmat
        pfor = pfor.at[:,:,n].set(pmat)
        
    return pfor

# pfor_true = torch.load(f'Output/pfor_true.pt')
# for i in range(50):
#     pfor = forward_propagation(b_prime)
#     # loss = criterion(pfor, pfor_true)
#     loss = (pfor.real-pfor_true.real)**2+(pfor.imag-pfor_true.imag)**2
#     loss.backward()
#     c -= 0.01*c.grad
pfor = forward_propagation(c, b_prime, Z, pfor, pref)
grad_forward_propagation = jax.grad(forward_propagation, argnums=0)
a_val = 3.0
gradient_at_c = grad_forward_propagation(c, b_prime, Z, pfor, pref)
print("The time difference is ", time.time() - start, "seconds")