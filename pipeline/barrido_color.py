"""
barrido_color.py
----------------
El corte superior W1-W2 < 1.60 se adopto porque 1.60 era un borde del binning que yo
use para partir la muestra por color, no porque se hubiera optimizado. Este script lo
barre entre 1.2 y sin corte, y responde una sola pregunta:

    ¿existe un umbral donde la componente perpendicular al CMB baje de forma
    ESTADISTICAMENTE SIGNIFICATIVA respecto a no cortar?

Si la respuesta es no, el corte es cosmetico y hay que reportarlo como tal.

DISENO

  * Configuracion fiducial: mascara A (region continua sobre rejilla NSIDE=512),
    rechazo de artefactos por referencia local, |b|>30, sin Nubes, W1<16.00,
    W1-W2 >= 0.80. Es la del resultado final.

  * Para cada umbral se mide el dipolo de la muestra RETENIDA y tambien el de las
    fuentes ELIMINADAS. Si el corte hace trabajo, lo eliminado debe tener un dipolo
    grande y apuntando lejos del CMB. Si lo eliminado es ruido, el corte no hace nada.

  * Las muestras estan ANIDADAS (bajar el umbral es un subconjunto del anterior), asi
    que las diferencias entre umbrales estan correlacionadas y compararlas con errores
    independientes exagera la significancia. Se usa BOOTSTRAP CONJUNTO: se remuestrea
    el catalogo padre una vez y se derivan todos los umbrales de ese mismo remuestreo.

  * El error incluye el factor 1.50x de inflacion por clustering medido en las
    simulaciones. Es una aproximacion: el factor se midio para la muestra completa y
    aqui se aplica a todos los umbrales.

Requiere: numpy, matplotlib, healpy, astropy.
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.table import Table

CACHE   = "cielo_cache.npz"
PARCHE  = "catwise_parche_profundo.ecsv"
NSIDE_ANA, NSIDE_FINA = 128, 512
B_CUT, RAD_MC = 30.0, 8.0
K_ART, FWHM_SUAVE = 8.0, 5.0
COLOR_MIN, W1_MAX, W1_PROFUNDO = 0.80, 16.00, 16.75

UMBRALES = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 99.0])   # 99 = sin corte
INFLA_CLUSTER = 1.50
N_BOOT = 300

V_SOBRE_C = 369.82/299792.458
CMB_L, CMB_B = 264.021, 48.253
LMC, SMC = (80.894, -69.756), (13.187, -72.829)
F0_W1, F0_W2, LAM_W1, LAM_W2 = 309.540, 171.787, 3.3526, 4.6028
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

def geom_de(lon, lat):
    ll, lb = eq2gal(*LMC); sl, sb = eq2gal(*SMC)
    return ((np.abs(lat) > B_CUT) & (sep_deg(lon, lat, ll, lb) > RAD_MC)
            & (sep_deg(lon, lat, sl, sb) > RAD_MC))

def centros(ns):
    th, ph = hp.pix2ang(ns, np.arange(hp.nside2npix(ns)))
    return np.degrees(ph), 90.0 - np.degrees(th)

def color_a_alpha(c):
    return -np.log((F0_W1/F0_W2)*10**(-c/2.5))/np.log(LAM_W2/LAM_W1)

def pendiente_local(mags, m0, ventana=0.30):
    mags = np.sort(mags[np.isfinite(mags)])
    mm = np.linspace(m0-ventana, m0+ventana, 20)
    NN = np.searchsorted(mags, mm); g = NN > 30
    return np.polyfit(mm[g], np.log10(NN[g]), 1)[0]/0.4 if g.sum() >= 5 else np.nan

# =============================== MASCARA ===============================
print("=" * 86)
print("1. CONFIGURACION FIDUCIAL")
z = np.load(CACHE)
ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
gl, gb = eq2gal(ra, dec); color = w1 - w2

NP_A = hp.nside2npix(NSIDE_ANA)
pix_a = hp.ang2pix(NSIDE_ANA, np.radians(90.0 - gb), np.radians(gl))
l_a, b_a = centros(NSIDE_ANA); geom_a = geom_de(l_a, b_a)
cnt_prof = np.bincount(pix_a[w1 < W1_PROFUNDO], minlength=NP_A).astype(float)
mf = geom_a.astype(float)
try:
    cs = hp.smoothing(cnt_prof*mf, fwhm=np.radians(FWHM_SUAVE), verbose=False)
    ms = hp.smoothing(mf, fwhm=np.radians(FWHM_SUAVE), verbose=False)
except TypeError:
    cs = hp.smoothing(cnt_prof*mf, fwhm=np.radians(FWHM_SUAVE))
    ms = hp.smoothing(mf, fwhm=np.radians(FWHM_SUAVE))
ref = np.where(ms > 0.2, cs/np.maximum(ms, 1e-6), np.nan)
exc = np.where(np.isfinite(ref) & (ref > 0), (cnt_prof-ref)/np.sqrt(np.maximum(ref, 1.0)), 0.0)
malas = geom_a & (exc > K_ART)

l_f, b_f = centros(NSIDE_FINA)
pix_f_a = hp.ang2pix(NSIDE_ANA, np.radians(90.0 - b_f), np.radians(l_f))
region = geom_de(l_f, b_f) & ~malas[pix_f_a]
u_mask = unit(l_f[region], b_f[region])
mu_r = u_mask.mean(0)
M = (u_mask.T @ u_mask)/len(u_mask) - np.outer(mu_r, mu_r)
Minv = np.linalg.inv(M)
n_cmb = unit(CMB_L, CMB_B)

padre = (color >= COLOR_MIN) & (w1 < W1_MAX) & geom_de(gl, gb) & ~malas[pix_a]
u_par = unit(gl[padre], gb[padre])
col_par = color[padre]
n_par = len(u_par)
print(f"   {malas.sum()} pixeles rechazados; muestra padre (sin corte superior): {n_par}")

tp = Table.read(PARCHE); cp = {c.lower(): c for c in tp.colnames}
w1p = np.asarray(tp[cp['w1mpro']], float)
colp = w1p - np.asarray(tp[cp['w2mpro']], float) if 'w2mpro' in cp else None

# cada fuente pertenece a los umbrales >= su indice
jmin = np.searchsorted(UMBRALES, col_par, side='left')

def dipolos(pesos):
    """Devuelve el dipolo RETENIDO y el ELIMINADO para cada umbral."""
    S = np.stack([np.bincount(jmin, weights=pesos*u_par[:, d], minlength=len(UMBRALES))
                  for d in range(3)], axis=1)
    N = np.bincount(jmin, weights=pesos, minlength=len(UMBRALES))
    Sc, Nc = np.cumsum(S, axis=0), np.cumsum(N)
    Stot, Ntot = S.sum(0), N.sum()
    ret = np.array([Minv @ (Sc[k]/Nc[k] - mu_r) for k in range(len(UMBRALES))])
    eli = np.array([Minv @ ((Stot-Sc[k])/(Ntot-Nc[k]) - mu_r) if Ntot-Nc[k] > 500
                    else np.full(3, np.nan) for k in range(len(UMBRALES))])
    return ret, eli, Nc, Ntot-Nc

# =============================== MEDICION ===============================
print("\n" + "=" * 86)
print("2. BARRIDO DEL UMBRAL SUPERIOR DE COLOR")
D_ret, D_eli, N_ret, N_eli = dipolos(np.ones(n_par))
par = D_ret @ n_cmb
perp = np.linalg.norm(D_ret - par[:, None]*n_cmb[None, :], axis=1)

print(f"\n   MUESTRA RETENIDA")
print(f"   {'umbral':>8} {'N':>9} {'%quit':>6} {'|D|':>9} {'l':>7} {'b':>7} "
      f"{'sep':>6} {'D_par':>9} {'D_perp':>9} {'x':>6} {'coc':>6}")
filas = []
for k, u in enumerate(UMBRALES):
    l, b = vec2lb(D_ret[k]); s = sep_deg(l, b, CMB_L, CMB_B)
    if colp is not None:
        mp = (colp >= COLOR_MIN) & (colp < u)
        xv = pendiente_local(w1p[mp], W1_MAX)
    else:
        xv = np.nan
    al = color_a_alpha(col_par[col_par < u].mean())
    De = (2 + xv*(1+al))*V_SOBRE_C
    et = "sin corte" if u > 90 else f"{u:.2f}"
    print(f"   {et:>8} {int(N_ret[k]):9d} {100*N_eli[k]/n_par:5.1f}% "
          f"{np.linalg.norm(D_ret[k]):9.5f} {l:7.1f} {b:+7.1f} {s:6.1f} "
          f"{par[k]:9.5f} {perp[k]:9.5f} {xv:6.3f} {np.linalg.norm(D_ret[k])/De:6.2f}")
    filas.append((u, N_ret[k], np.linalg.norm(D_ret[k]), s, par[k], perp[k], De))
filas = np.array(filas)

print(f"\n   FUENTES ELIMINADAS  (si el corte hace trabajo, esto debe apuntar lejos del CMB)")
print(f"   {'umbral':>8} {'N':>9} {'|D|':>9} {'l':>7} {'b':>7} {'sep':>6}")
for k, u in enumerate(UMBRALES):
    if not np.isfinite(D_eli[k]).all() or N_eli[k] < 500:
        continue
    l, b = vec2lb(D_eli[k])
    print(f"   {u:8.2f} {int(N_eli[k]):9d} {np.linalg.norm(D_eli[k]):9.5f} {l:7.1f} "
          f"{b:+7.1f} {sep_deg(l,b,CMB_L,CMB_B):6.1f}")

# =============================== BOOTSTRAP CONJUNTO ===============================
print("\n" + "=" * 86)
print(f"3. BOOTSTRAP CONJUNTO ({N_BOOT} remuestreos del MISMO catalogo)")
print("   Los umbrales estan anidados: compararlos con errores independientes")
print("   exageraria la significancia. Aqui las diferencias llevan su correlacion.\n")
rng = np.random.default_rng(7)
b_perp = np.empty((N_BOOT, len(UMBRALES)))
b_sep = np.empty((N_BOOT, len(UMBRALES)))
for i in range(N_BOOT):
    pes = np.bincount(rng.integers(0, n_par, n_par), minlength=n_par).astype(float)
    Dr, _, _, _ = dipolos(pes)
    p = Dr @ n_cmb
    b_perp[i] = np.linalg.norm(Dr - p[:, None]*n_cmb[None, :], axis=1)*INFLA_CLUSTER
    for k in range(len(UMBRALES)):
        l, b = vec2lb(Dr[k]); b_sep[i, k] = sep_deg(l, b, CMB_L, CMB_B)

sd_perp = b_perp.std(0)
k_sin = len(UMBRALES)-1
d_perp = perp - perp[k_sin]
sd_dif = (b_perp - b_perp[:, [k_sin]]).std(0)

print(f"   {'umbral':>10} {'D_perp':>9} {'+/-':>8} {'sep':>7} {'+/-':>6} "
      f"{'perp - sin corte':>18} {'sigma':>7}")
for k, u in enumerate(UMBRALES):
    et = "sin corte" if u > 90 else f"{u:.2f}"
    sg = d_perp[k]/sd_dif[k] if sd_dif[k] > 0 else 0.0
    print(f"   {et:>10} {perp[k]:9.5f} {sd_perp[k]:8.5f} {filas[k,3]:7.1f} "
          f"{b_sep[:,k].std():6.1f} {d_perp[k]:+18.5f} {sg:+7.2f}")

# =============================== VEREDICTO ===============================
print("\n" + "=" * 86)
print("4. VEREDICTO")
mejor = int(np.nanargmin(perp))
sig_mejor = d_perp[mejor]/sd_dif[mejor] if sd_dif[mejor] > 0 else 0.0
print(f"   umbral con menor D_perp: {UMBRALES[mejor]:.2f}  "
      f"(D_perp = {perp[mejor]:.5f})")
print(f"   diferencia frente a no cortar: {d_perp[mejor]:+.5f} = {sig_mejor:+.2f} sigma")
print(f"\n   rango de D_perp en todo el barrido: {perp.min():.5f} a {perp.max():.5f}")
print(f"   error tipico de D_perp (con clustering): {sd_perp.mean():.5f}")
if abs(sig_mejor) < 2.0:
    print("\n   >>> NINGUN umbral mejora de forma significativa (todo por debajo de 2 sigma).")
    print("   El corte superior de color es COSMETICO. Lo honesto es o bien no aplicarlo,")
    print("   o aplicarlo declarando que su efecto esta dentro del ruido y que el valor")
    print("   1.60 procede de un borde de binning, no de una optimizacion.")
else:
    print(f"\n   >>> El umbral {UMBRALES[mejor]:.2f} mejora {abs(sig_mejor):.1f} sigma.")
    print("   Eso SI es una calibracion. Reportalo con este barrido como justificacion,")
    print("   y advierte que el umbral se eligio minimizando D_perp sobre 7 opciones.")

# =============================== FIGURA ===============================
uu = np.where(UMBRALES > 90, 2.8, UMBRALES)
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5), dpi=145)

axs[0].errorbar(uu, perp*100, yerr=sd_perp*100, fmt='o-', color='#b5423a', lw=2, ms=7, capsize=4)
axs[0].axhline(perp[k_sin]*100, color='k', ls=':', lw=1.5, label='sin corte')
axs[0].axvline(1.60, color='#2e6fa8', ls='--', lw=1.5, label='1.60 adoptado')
axs[0].set_xlabel("umbral superior W1−W2"); axs[0].set_ylabel("componente ⊥ al CMB (%)")
axs[0].set_title("¿Baja el contaminante?"); axs[0].legend(fontsize=8.5); axs[0].grid(alpha=.3)

axs[1].errorbar(uu, filas[:, 3], yerr=b_sep.std(0), fmt='o-', color='#c77a30',
                lw=2, ms=7, capsize=4)
axs[1].axhline(0, color='k', ls=':', lw=1.5, label='dirección del CMB')
axs[1].axvline(1.60, color='#2e6fa8', ls='--', lw=1.5)
axs[1].set_xlabel("umbral superior W1−W2"); axs[1].set_ylabel("separación del CMB (grados)")
axs[1].set_title("¿Se estabiliza la dirección?"); axs[1].legend(fontsize=8.5); axs[1].grid(alpha=.3)

ok = np.isfinite(D_eli).all(axis=1) & (N_eli > 500)
if ok.any():
    axs[2].plot(uu[ok], np.linalg.norm(D_eli[ok], axis=1)*100, 's-', color='#1b6b50',
                lw=2, ms=7, label='|D| de lo eliminado')
    ax2b = axs[2].twinx()
    ax2b.plot(uu[ok], [sep_deg(*vec2lb(D_eli[k]), CMB_L, CMB_B) for k in np.where(ok)[0]],
              '^--', color='#7a4fa3', lw=1.8, ms=6, label='separación del CMB')
    ax2b.set_ylabel("separación del CMB (grados)")
    axs[2].set_xlabel("umbral superior W1−W2"); axs[2].set_ylabel("|D| eliminado (%)")
    axs[2].set_title("Lo que el corte tira a la basura")
    h1, l1 = axs[2].get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    axs[2].legend(h1+h2, l1+l2, fontsize=8); axs[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig("barrido_color.png", dpi=145)
print("\nFigura guardada: barrido_color.png")
plt.show()

print("""
COMO LEERLO

 * Panel 1. Si D_perp baja y luego se aplana, el corte hace trabajo y el codo marca
   el umbral. Si es plano dentro de las barras, no hace nada.
 * Panel 2. Lo mismo con la direccion. Ojo: la separacion angular NO es simetrica ni
   gaussiana cerca de cero, asi que su barra es orientativa; la cantidad rigurosa es
   D_perp del panel 1.
 * Panel 3. El dipolo de las fuentes eliminadas. Si es grande y apunta lejos del CMB,
   lo que se quita es contaminacion. Si es pequeño o apunta a cualquier lado, lo que
   se quita es ruido y el corte solo resta estadistica.

 Y la advertencia de siempre: elegir el umbral que minimiza D_perp entre 7 opciones
 es exactamente el jardin de senderos que se bifurcan. Solo vale como calibracion si
 la mejora supera holgadamente el ruido; si no, el resultado del barrido es que el
 corte no esta justificado por los datos.
""")
