"""
clustering_v3.py
----------------
Segunda correccion. El borde ya esta arreglado (C_2 bajo de 11.7x a 1.4x el ruido de
disparo), pero el check contra el estimador de puntos sigue fallando en +8.9%.

EL ERROR RESTANTE
  El rechazo de artefactos compara cada pixel contra la MEDIA GLOBAL del mapa:
  umbral = 107.2 + 12*sqrt(107.2) = 231. Eso rechazo 1324 pixeles, el 5.5% de la
  mascara, contra los 32 que daba la misma configuracion en reconciliar.py.

  Una media global supone densidad de fondo uniforme. No lo es: varia con la latitud
  galactica al acercarse al plano, con la ecliptica por la cobertura, y el clustering
  hace que la varianza de conteos en celdas sea bastante mayor que Poisson. Con
  referencia global, regiones legitimamente densas se marcan como artefactos; si esas
  regiones forman una banda o un anillo, vuelves a meter estructura en el mapa.

LA CORRECCION
  Referencia LOCAL. Se suaviza el mapa de conteos con un haz de unos grados y cada
  pixel se compara contra su propio entorno:

      exceso_i = (c_i - c_suave_i) / sqrt(c_suave_i)

  Un artefacto real es una concentracion puntual muy por encima de su vecindad; una
  region densa por gradiente galactico o por clustering sube junto con su entorno y
  no se marca. El script imprime ademas DONDE caen los pixeles rechazados, para que
  se vea si el criterio viejo estaba trazando un anillo.

Requiere: numpy, matplotlib, healpy, scipy.
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy import stats

CACHE = "cielo_cache.npz"
NSIDE = 64
B_CUT, RAD_MC = 30.0, 8.0
K_ART = 8.0                  # ahora en unidades de sigma LOCAL
FWHM_SUAVE = 5.0             # grados, escala del entorno para la referencia local
COLOR_MIN, COLOR_MAX, W1_MAX = 0.80, 1.60, 16.00
W1_PROFUNDO = 16.75
D_ESPERADO = 0.00670
N_MOCKS = 2000
LMAX = 3*NSIDE - 1
REF_PUNTOS = 0.01389

CMB_L, CMB_B = 264.021, 48.253
LMC, SMC = (80.894, -69.756), (13.187, -72.829)
EPS = np.radians(23.4392911)
RA_NGP, DEC_NGP, L_NCP = np.radians(192.85948), np.radians(27.12825), np.radians(122.93192)

def eq2gal(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    b = np.arcsin(np.clip(np.sin(dec)*np.sin(DEC_NGP) +
                          np.cos(dec)*np.cos(DEC_NGP)*np.cos(ra-RA_NGP), -1, 1))
    y = np.cos(dec)*np.sin(ra-RA_NGP)
    x = np.sin(dec)*np.cos(DEC_NGP) - np.cos(dec)*np.sin(DEC_NGP)*np.cos(ra-RA_NGP)
    return np.degrees(L_NCP - np.arctan2(y, x)) % 360, np.degrees(b)

def gal2eq(l, b):
    l, b = np.radians(l), np.radians(b)
    sd = np.sin(b)*np.sin(DEC_NGP) + np.cos(b)*np.cos(DEC_NGP)*np.cos(L_NCP-l)
    dec = np.arcsin(np.clip(sd, -1, 1))
    y = np.cos(b)*np.sin(L_NCP-l)
    x = np.sin(b)*np.cos(DEC_NGP) - np.cos(b)*np.sin(DEC_NGP)*np.cos(L_NCP-l)
    return np.degrees(RA_NGP + np.arctan2(y, x)) % 360, np.degrees(dec)

def eq2ecl(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.degrees(np.arcsin(np.clip(np.sin(dec)*np.cos(EPS) -
                                        np.cos(dec)*np.sin(EPS)*np.sin(ra), -1, 1)))

def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], axis=-1)

def vec2lb(v):
    n = v/np.linalg.norm(v)
    return np.degrees(np.arctan2(n[1], n[0])) % 360, np.degrees(np.arcsin(np.clip(n[2], -1, 1)))

def sep_deg(l1, b1, l2, b2):
    return np.degrees(np.arccos(np.clip(np.sum(unit(l1, b1)*unit(l2, b2), axis=-1), -1, 1)))

# =============================== 1. GEOMETRIA ===============================
print("=" * 76)
print("1. MASCARA")
z = np.load(CACHE)
ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
gl, gb = eq2gal(ra, dec); color = w1 - w2

NPIX = hp.nside2npix(NSIDE)
pix_src = hp.ang2pix(NSIDE, np.radians(90.0 - gb), np.radians(gl))
th, ph = hp.pix2ang(NSIDE, np.arange(NPIX))
l_pix, b_pix = np.degrees(ph), 90.0 - np.degrees(th)
u_pix = unit(l_pix, b_pix)
ra_pix, dec_pix = gal2eq(l_pix, b_pix)
be_pix = eq2ecl(ra_pix, dec_pix)

lmc_l, lmc_b = eq2gal(*LMC); smc_l, smc_b = eq2gal(*SMC)
geom = ((np.abs(b_pix) > B_CUT)
        & (sep_deg(l_pix, b_pix, lmc_l, lmc_b) > RAD_MC)
        & (sep_deg(l_pix, b_pix, smc_l, smc_b) > RAD_MC))
print(f"   mascara geometrica (solo centros de pixel): {geom.sum()} pixeles, "
      f"f_sky = {geom.mean():.3f}")

# =============================== 2. RECHAZO LOCAL ===============================
print("\n" + "=" * 76)
print("2. RECHAZO DE ARTEFACTOS CON REFERENCIA LOCAL")
cnt_prof = np.bincount(pix_src[w1 < W1_PROFUNDO], minlength=NPIX).astype(float)

# suavizado dentro de la mascara: se suavizan conteos y mascara, y se divide
m_f = geom.astype(float)
c_s = hp.smoothing(cnt_prof*m_f, fwhm=np.radians(FWHM_SUAVE), verbose=False) \
      if 'verbose' in hp.smoothing.__code__.co_varnames else \
      hp.smoothing(cnt_prof*m_f, fwhm=np.radians(FWHM_SUAVE))
m_s = hp.smoothing(m_f, fwhm=np.radians(FWHM_SUAVE), verbose=False) \
      if 'verbose' in hp.smoothing.__code__.co_varnames else \
      hp.smoothing(m_f, fwhm=np.radians(FWHM_SUAVE))
ref = np.where(m_s > 0.2, c_s/np.maximum(m_s, 1e-6), np.nan)

exceso = np.where(np.isfinite(ref) & (ref > 0),
                  (cnt_prof - ref)/np.sqrt(np.maximum(ref, 1.0)), 0.0)
malas = geom & (exceso > K_ART)
mask = geom & ~malas
print(f"   referencia local: haz de {FWHM_SUAVE} grados")
print(f"   rango de la referencia dentro de la mascara: "
      f"{np.nanmin(ref[geom]):.0f} a {np.nanmax(ref[geom]):.0f} fuentes/pixel "
      f"(media {np.nanmean(ref[geom]):.0f})")
print(f"   -> la densidad de fondo varia un factor "
      f"{np.nanmax(ref[geom])/np.nanmin(ref[geom]):.1f} sobre la mascara;")
print(f"      por eso una media global no sirve como referencia")
print(f"\n   {malas.sum()} pixeles rechazados a {K_ART} sigma LOCAL")
print(f"   [criterio global anterior: 1324 pixeles; reconciliar.py A3: 32]")

# ¿donde caen los rechazados por cada criterio?
mu_glob = cnt_prof[geom & (cnt_prof > 0)].mean()
malas_glob = geom & (cnt_prof > mu_glob + 12.0*np.sqrt(mu_glob))
print(f"\n   DONDE CAEN LOS RECHAZADOS  (distribucion en |b| galactica)")
print(f"      {'|b|':>10} {'pixeles mascara':>16} {'global':>9} {'local':>8}")
for lo in range(30, 90, 10):
    s = geom & (np.abs(b_pix) >= lo) & (np.abs(b_pix) < lo+10)
    print(f"      {lo:3d}-{lo+10:3d}  {s.sum():16d} {(malas_glob & s).sum():9d} "
          f"{(malas & s).sum():8d}")
print(f"\n      {'|beta_ecl|':>10} {'pixeles mascara':>16} {'global':>9} {'local':>8}")
for lo in range(0, 90, 15):
    s = geom & (np.abs(be_pix) >= lo) & (np.abs(be_pix) < lo+15)
    print(f"      {lo:3d}-{lo+15:3d}  {s.sum():16d} {(malas_glob & s).sum():9d} "
          f"{(malas & s).sum():8d}")
print("\n   Si el criterio global concentra sus rechazos en una banda y el local no,")
print("   el global estaba borrando estructura real y metiendo un patron en el mapa.")

# =============================== 3. DIPOLO ===============================
cien = (color >= COLOR_MIN) & (color < COLOR_MAX) & (w1 < W1_MAX)
conteo = np.bincount(pix_src[cien], minlength=NPIX).astype(float)
f_sky = mask.mean(); n_tot = conteo[mask].sum(); n_bar = n_tot/mask.sum()

def dipolo(c, m):
    uu = u_pix[m]; cc = c[m]
    mu_d = (cc[:, None]*uu).sum(0)/cc.sum()
    mu_r = uu.mean(0)
    M = (uu.T @ uu)/len(uu) - np.outer(mu_r, mu_r)
    return np.linalg.solve(M, mu_d - mu_r)

print("\n" + "=" * 76)
print("3. DIPOLO")
D_obs = dipolo(conteo, mask)
a_obs = np.linalg.norm(D_obs); l_obs, b_obs = vec2lb(D_obs)
n_cmb = unit(CMB_L, CMB_B); par_obs = D_obs @ n_cmb
print(f"   N = {int(n_tot)} fuentes, f_sky = {f_sky:.3f}, {n_bar:.1f} por pixel")
print(f"   |D| = {a_obs:.5f}   (l,b) = ({l_obs:.1f}, {b_obs:+.1f})   "
      f"sep = {sep_deg(l_obs,b_obs,CMB_L,CMB_B):.1f} grados")
d_rel = 100*(a_obs - REF_PUNTOS)/REF_PUNTOS
print(f"\n   CHECK contra el estimador de puntos ({REF_PUNTOS:.5f}): {d_rel:+.1f}%")
print(f"   {'OK, consistente' if abs(d_rel) < 5 else '*** AUN NO CIERRA ***'}")
print(f"   [v1 con borde malo: +33% ; v2 con borde bueno y umbral global: +8.9%]")

# =============================== 4. ESPECTRO ===============================
print("\n" + "=" * 76)
print("4. ESPECTRO")
delta = np.zeros(NPIX); delta[mask] = conteo[mask]/n_bar - 1.0
cl_obs = hp.anafast(delta*mask, lmax=LMAX)/f_sky
shot = (4*np.pi/NPIX)/n_bar
cl_sig = np.maximum(cl_obs - shot, 0.0); cl_sig[0:2] = 0.0
print(f"   C_2 = {cl_sig[2]:.3e}  ({cl_sig[2]/shot:.1f}x el ruido de disparo)")
print(f"   [v1: 11.7x (borde) ; v2: 1.4x]")

# =============================== 5. SIMULACIONES ===============================
def simular(n, D_iny=None, semilla=0):
    r = np.random.default_rng(semilla); out = np.empty((n, 3))
    for i in range(n):
        try:
            d = hp.synfast(cl_sig, NSIDE, lmax=LMAX, verbose=False)
        except TypeError:
            d = hp.synfast(cl_sig, NSIDE, lmax=LMAX)
        if D_iny is not None:
            d = d + u_pix @ D_iny
        out[i] = dipolo(r.poisson(np.clip(n_bar*(1.0+d), 0, None)).astype(float), mask)
    return out

print("\n" + "=" * 76)
print("5. VALIDACION Y SIGNIFICANCIA")
D_null = simular(500, None, semilla=1)
sig_c = D_null.std(0).mean(); sig_p = np.sqrt(3.0/n_tot)
print(f"   sigma/componente: Poisson {sig_p:.5f} -> clustering {sig_c:.5f} "
      f"(factor {sig_c/sig_p:.2f}x)   [v1 1.81x, v2 1.49x]")
print(f"   recuperacion:")
for amp in (0.005, 0.010, 0.015):
    Dr = simular(200, amp*n_cmb, semilla=int(amp*1e5)) @ n_cmb
    print(f"      {amp:.4f} -> {Dr.mean():.5f} +/- {Dr.std():.5f} "
          f"({100*(Dr.mean()-amp)/amp:+.1f}%)")

D_kin = simular(N_MOCKS, D_ESPERADO*n_cmb, semilla=2026)
a_kin = np.linalg.norm(D_kin, axis=1); par_kin = D_kin @ n_cmb
s_a = (a_obs - a_kin.mean())/a_kin.std()
s_p = (par_obs - par_kin.mean())/par_kin.std()
print(f"\n   NULA CINEMATICA (D = {D_ESPERADO:.5f}), {N_MOCKS} simulaciones")
print(f"      |D|      : {a_obs:.5f} contra {a_kin.mean():.5f} +/- {a_kin.std():.5f}"
      f"   {(a_kin >= a_obs).sum()}/{N_MOCKS}  ->  {s_a:.2f} sigma")
print(f"      paralela : {par_obs:.5f} contra {par_kin.mean():.5f} +/- {par_kin.std():.5f}"
      f"   {(par_kin >= par_obs).sum()}/{N_MOCKS}  ->  {s_p:.2f} sigma")
print(f"\n   COCIENTE: {a_obs/D_ESPERADO:.2f} +/- {a_kin.std()/D_ESPERADO:.2f}")

# =============================== 6. FIGURA ===============================
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5), dpi=145)
bb = np.arange(30, 90, 5)
g_glob = [(malas_glob & geom & (np.abs(b_pix) >= x) & (np.abs(b_pix) < x+5)).sum() /
          max((geom & (np.abs(b_pix) >= x) & (np.abs(b_pix) < x+5)).sum(), 1) for x in bb]
g_loc = [(malas & geom & (np.abs(b_pix) >= x) & (np.abs(b_pix) < x+5)).sum() /
         max((geom & (np.abs(b_pix) >= x) & (np.abs(b_pix) < x+5)).sum(), 1) for x in bb]
axs[0].plot(bb+2.5, np.array(g_glob)*100, 'o-', color='#b5423a', lw=2, label='umbral global')
axs[0].plot(bb+2.5, np.array(g_loc)*100, 's-', color='#1b6b50', lw=2, label='referencia local')
axs[0].set_xlabel("|b| galáctica"); axs[0].set_ylabel("% de píxeles rechazados")
axs[0].set_title("¿Dónde rechaza cada criterio?"); axs[0].legend(fontsize=8.5); axs[0].grid(alpha=.3)

ll = np.arange(2, min(64, len(cl_sig)))
axs[1].semilogy(ll, cl_obs[ll], 'o-', color='0.55', ms=3, lw=1, label='medido')
axs[1].semilogy(ll, np.maximum(cl_sig[ll], 1e-12), 'o-', color='#b5423a', ms=3, lw=1.5,
                label='señal')
axs[1].axhline(shot, color='k', ls=':', lw=1.5, label='ruido de disparo')
axs[1].set_xlabel("multipolo ℓ"); axs[1].set_ylabel(r"$C_\ell$ (sr)")
axs[1].set_title("Espectro"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)

axs[2].hist(par_kin*100, bins=50, color='#1b6b50', alpha=.7)
axs[2].axvline(par_obs*100, color='#b5423a', lw=2.5, label=f'observado {par_obs*100:.2f}%')
axs[2].axvline(D_ESPERADO*100, color='k', ls='--', lw=1.5, label='esperado')
axs[2].set_xlabel("componente ∥ al CMB (%)"); axs[2].set_ylabel("simulaciones")
axs[2].set_title(f"Componente paralela  ({s_p:.1f}σ)")
axs[2].legend(fontsize=8); axs[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig("clustering_v3.png", dpi=145)
print("\nFigura guardada: clustering_v3.png")
plt.show()

print("""
ORDEN DE LECTURA

 1. El CHECK del apartado 3. Si no cae dentro del 5%, el resto es provisional.
 2. La tabla de DONDE CAEN LOS RECHAZADOS. Es lo que explica por que el umbral global
    fallaba, y de paso te dice si el rechazo de artefactos que usaste en todo el
    estudio anterior tenia el mismo defecto.
 3. Solo entonces, la significancia.

 K_ART = 8 sigma LOCAL no es lo mismo que 12 sigma sobre la media global: la
 referencia local ya absorbe la variacion de fondo, asi que el umbral puede ser mas
 estricto sin borrar estructura legitima. Si rechaza demasiado o demasiado poco,
 muevelo y mira si el dipolo se mueve con el; si se mueve mucho, ese es un sistematico
 mas que hay que reportar.
""")
