"""
clustering_v2.py
----------------
Corrige el error de borde diagnosticado por reconciliar.py.

EL ERROR
  En clustering.py las fuentes se seleccionaban con |b|>30 FUENTE POR FUENTE, pero la
  mascara se evaluaba con |b|>30 en el CENTRO DEL PIXEL. Un pixel con centro en
  b=30.2 entraba entero en la mascara pero solo estaba poblado por encima de b=30.
  Eso deja un anillo de deficit artificial en el borde, simetrico respecto al plano
  galactico, que se acopla al dipolo en un cielo cortado. Efecto medido: +31% en la
  amplitud, con 6418 fuentes perdidas en el borde.

LA CORRECCION
  TODO a nivel de pixel. Las fuentes no se filtran por su propia posicion galactica:
  se asignan a pixeles y se conserva el pixel entero o se descarta entero. Asi cada
  pixel de la mascara esta poblado en toda su area. Lo mismo para las Nubes de
  Magallanes.

  Los cortes de COLOR y MAGNITUD si son por fuente: son propiedades del objeto, no
  de su posicion, y no generan bordes geometricos.

  El rechazo de artefactos se calcula sobre la muestra PROFUNDA (W1<16.75), donde los
  artefactos destacan mas sobre el ruido de Poisson, y se aplica a la de ciencia.
  En reconciliar.py esa mascara (A3) dio 0.01370 a 1.4 grados del CMB, consistente
  con el estimador de puntos.

Requiere: numpy, matplotlib, healpy, scipy.
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy import stats

CACHE = "cielo_cache.npz"
NSIDE = 64
B_CUT, RAD_MC, K_ART = 30.0, 8.0, 12.0
COLOR_MIN, COLOR_MAX, W1_MAX = 0.80, 1.60, 16.00
W1_PROFUNDO = 16.75
D_ESPERADO = 0.00670
N_MOCKS = 2000
LMAX = 3*NSIDE - 1

CMB_L, CMB_B = 264.021, 48.253
LMC, SMC = (80.894, -69.756), (13.187, -72.829)

RA_NGP, DEC_NGP, L_NCP = np.radians(192.85948), np.radians(27.12825), np.radians(122.93192)

def eq2gal(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    b = np.arcsin(np.clip(np.sin(dec)*np.sin(DEC_NGP) +
                          np.cos(dec)*np.cos(DEC_NGP)*np.cos(ra-RA_NGP), -1, 1))
    y = np.cos(dec)*np.sin(ra-RA_NGP)
    x = np.sin(dec)*np.cos(DEC_NGP) - np.cos(dec)*np.sin(DEC_NGP)*np.cos(ra-RA_NGP)
    return np.degrees(L_NCP - np.arctan2(y, x)) % 360, np.degrees(b)

def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], axis=-1)

def vec2lb(v):
    n = v/np.linalg.norm(v)
    return np.degrees(np.arctan2(n[1], n[0])) % 360, np.degrees(np.arcsin(np.clip(n[2], -1, 1)))

def sep_deg(l1, b1, l2, b2):
    return np.degrees(np.arccos(np.clip(np.sum(unit(l1, b1)*unit(l2, b2), axis=-1), -1, 1)))

# =============================== 1. MASCARA, TODA POR PIXEL ===============================
print("=" * 74)
print("1. MASCARA DEFINIDA ENTERAMENTE A NIVEL DE PIXEL")
z = np.load(CACHE)
ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
gl, gb = eq2gal(ra, dec); color = w1 - w2

NPIX = hp.nside2npix(NSIDE)
pix_src = hp.ang2pix(NSIDE, np.radians(90.0 - gb), np.radians(gl))
th, ph = hp.pix2ang(NSIDE, np.arange(NPIX))
l_pix, b_pix = np.degrees(ph), 90.0 - np.degrees(th)
u_pix = unit(l_pix, b_pix)

lmc_l, lmc_b = eq2gal(*LMC); smc_l, smc_b = eq2gal(*SMC)
# geometria: SOLO en centros de pixel. Las fuentes no se filtran por su propia b.
mask = ((np.abs(b_pix) > B_CUT)
        & (sep_deg(l_pix, b_pix, lmc_l, lmc_b) > RAD_MC)
        & (sep_deg(l_pix, b_pix, smc_l, smc_b) > RAD_MC))

# cortes por fuente: color y magnitud (propiedades del objeto, no de la posicion)
prof_src = w1 < W1_PROFUNDO
cien_src = (color >= COLOR_MIN) & (color < COLOR_MAX) & (w1 < W1_MAX)

# rechazo de artefactos sobre la muestra PROFUNDA
cnt_prof = np.bincount(pix_src[prof_src], minlength=NPIX).astype(float)
ocup = cnt_prof[mask & (cnt_prof > 0)]
mu_p = ocup.mean()
malas = cnt_prof > mu_p + K_ART*np.sqrt(mu_p)
mask &= ~malas
print(f"   {malas.sum()} pixeles rechazados por artefactos "
      f"(umbral {mu_p:.1f} + {K_ART}*sqrt, calculado en W1<{W1_PROFUNDO})")

conteo = np.bincount(pix_src[cien_src], minlength=NPIX).astype(float)
f_sky = mask.mean()
n_tot = conteo[mask].sum()
n_bar = n_tot/mask.sum()
print(f"   N = {int(n_tot)} fuentes en {mask.sum()} pixeles  (f_sky = {f_sky:.3f})")
print(f"   densidad media = {n_bar:.1f} fuentes/pixel")

def dipolo(c, m):
    uu = u_pix[m]; cc = c[m]
    mu_d = (cc[:, None]*uu).sum(0)/cc.sum()
    mu_r = uu.mean(0)
    M = (uu.T @ uu)/len(uu) - np.outer(mu_r, mu_r)
    return np.linalg.solve(M, mu_d - mu_r)

D_obs = dipolo(conteo, mask)
a_obs = np.linalg.norm(D_obs); l_obs, b_obs = vec2lb(D_obs)
n_cmb = unit(CMB_L, CMB_B); par_obs = D_obs @ n_cmb
print(f"\n   DIPOLO OBSERVADO: |D| = {a_obs:.5f}   (l,b) = ({l_obs:.1f}, {b_obs:+.1f})")
print(f"   separacion del CMB = {sep_deg(l_obs,b_obs,CMB_L,CMB_B):.1f} grados")
print(f"   componente paralela = {par_obs:.5f}")
print(f"\n   CHECK: el estimador de puntos da 0.01389 a 1.4 grados.")
d_rel = 100*(a_obs - 0.01389)/0.01389
print(f"   diferencia: {d_rel:+.1f}%  -> {'OK, consistente' if abs(d_rel) < 5 else '*** SIGUE HABIENDO UN PROBLEMA ***'}")

# =============================== 2. ESPECTRO ===============================
print("\n" + "=" * 74)
print("2. ESPECTRO DE POTENCIAS")
delta = np.zeros(NPIX)
delta[mask] = conteo[mask]/n_bar - 1.0
cl_obs = hp.anafast(delta*mask, lmax=LMAX)/f_sky
omega_pix = 4*np.pi/NPIX
shot = omega_pix/n_bar
cl_sig = np.maximum(cl_obs - shot, 0.0)
cl_sig[0:2] = 0.0
print(f"   ruido de disparo = {shot:.3e} sr")
print(f"   C_2 = {cl_sig[2]:.3e}  ({cl_sig[2]/shot:.1f}x el ruido de disparo)")
print(f"   [con la mascara defectuosa daba 9.659e-05, o sea 11.7x]")
print(f"   C_l:  l=5: {cl_sig[5]:.3e}   l=10: {cl_sig[10]:.3e}   l=30: {cl_sig[30]:.3e}")

# =============================== 3. SIMULACIONES ===============================
def simular(n, D_iny=None, semilla=0):
    r = np.random.default_rng(semilla)
    out = np.empty((n, 3))
    for i in range(n):
        try:
            d = hp.synfast(cl_sig, NSIDE, lmax=LMAX, verbose=False)
        except TypeError:
            d = hp.synfast(cl_sig, NSIDE, lmax=LMAX)
        if D_iny is not None:
            d = d + u_pix @ D_iny
        lam = np.clip(n_bar*(1.0 + d), 0, None)
        out[i] = dipolo(r.poisson(lam).astype(float), mask)
    return out

print("\n" + "=" * 74)
print("3. VALIDACION")
D_null = simular(500, None, semilla=1)
sig_clust = D_null.std(0).mean()
sig_pois = np.sqrt(3.0/n_tot)
print(f"   nulas: media vectorial {np.linalg.norm(D_null.mean(0)):.6f} (debe ser ~0)")
print(f"   sigma por componente: Poisson {sig_pois:.5f} -> con clustering {sig_clust:.5f}"
      f"  (factor {sig_clust/sig_pois:.2f}x)")
print(f"   [con la mascara defectuosa el factor era 1.81x]")

print(f"\n   recuperacion de dipolos inyectados:")
for amp in (0.005, 0.010, 0.015):
    Dr = simular(200, amp*n_cmb, semilla=int(amp*1e5)) @ n_cmb
    print(f"      {amp:.4f} -> {Dr.mean():.5f} +/- {Dr.std():.5f}  "
          f"({100*(Dr.mean()-amp)/amp:+.1f}%)")

# =============================== 4. SIGNIFICANCIA ===============================
print("\n" + "=" * 74)
print(f"4. HIPOTESIS NULA: dipolo puramente cinematico (D = {D_ESPERADO:.5f})")
D_kin = simular(N_MOCKS, D_ESPERADO*n_cmb, semilla=2026)
a_kin = np.linalg.norm(D_kin, axis=1); par_kin = D_kin @ n_cmb

n_exc = (a_kin >= a_obs).sum()
sig_a = (a_obs - a_kin.mean())/a_kin.std()
n_exc_p = (par_kin >= par_obs).sum()
sig_p = (par_obs - par_kin.mean())/par_kin.std()

print(f"\n   |D|: observado {a_obs:.5f} contra {a_kin.mean():.5f} +/- {a_kin.std():.5f}")
print(f"        {n_exc} de {N_MOCKS} lo superan  ->  {sig_a:.2f} sigma")
print(f"   paralela: observada {par_obs:.5f} contra {par_kin.mean():.5f} +/- {par_kin.std():.5f}")
print(f"        {n_exc_p} de {N_MOCKS} la superan  ->  {sig_p:.2f} sigma")
print(f"\n   COCIENTE: {a_obs/D_ESPERADO:.2f} +/- {a_kin.std()/D_ESPERADO:.2f}")
print(f"   [Secrest et al. 2021: 4.9 sigma ; reevaluacion 2025: 3.3 sigma]")

# =============================== 5. FIGURA ===============================
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5), dpi=145)
ll = np.arange(2, min(64, len(cl_sig)))
axs[0].semilogy(ll, cl_obs[ll], 'o-', color='0.55', ms=3, lw=1, label='medido')
axs[0].semilogy(ll, np.maximum(cl_sig[ll], 1e-12), 'o-', color='#b5423a', ms=3, lw=1.5,
                label='señal')
axs[0].axhline(shot, color='k', ls=':', lw=1.5, label='ruido de disparo')
axs[0].set_xlabel("multipolo ℓ"); axs[0].set_ylabel(r"$C_\ell$ (sr)")
axs[0].set_title("Espectro, máscara corregida"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)

axs[1].hist(a_kin*100, bins=50, color='#2e6fa8', alpha=.7)
axs[1].axvline(a_obs*100, color='#b5423a', lw=2.5, label=f'observado {a_obs*100:.2f}%')
axs[1].axvline(D_ESPERADO*100, color='k', ls='--', lw=1.5, label='esperado')
axs[1].set_xlabel("|D| (%)"); axs[1].set_ylabel("simulaciones")
axs[1].set_title(f"Nula cinemática  ({sig_a:.1f}σ)"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)

axs[2].hist(par_kin*100, bins=50, color='#1b6b50', alpha=.7)
axs[2].axvline(par_obs*100, color='#b5423a', lw=2.5, label=f'observado {par_obs*100:.2f}%')
axs[2].axvline(D_ESPERADO*100, color='k', ls='--', lw=1.5, label='esperado')
axs[2].set_xlabel("componente ∥ al CMB (%)"); axs[2].set_ylabel("simulaciones")
axs[2].set_title(f"Componente paralela  ({sig_p:.1f}σ)"); axs[2].legend(fontsize=8); axs[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig("clustering_v2.png", dpi=145)
print("\nFigura guardada: clustering_v2.png")
plt.show()

print("""
LO PRIMERO QUE HAY QUE MIRAR es el CHECK del apartado 1. Si la amplitud no cae dentro
del 5% de 0.01389, la correccion no fue suficiente y no tiene sentido leer el resto.

Si el check pasa, los dos numeros que importan son:
  * el factor de inflacion del error por clustering (antes salio 1.81x con el mapa
    contaminado; ahora deberia bajar, porque parte de aquel C_2 era el anillo de borde)
  * la significancia sobre la componente PARALELA, que es la cantidad fisica limpia

Y la advertencia de siempre: la correccion del C_l por f_sky sigue siendo cruda, asi
que estos sigmas tienen una incertidumbre de metodo que no esta en la barra.
""")
