"""
estabilidad_corte.py
--------------------
EL TEST DECISIVO.

Hasta ahora mediste D_observado con un corte (W1<16.5) y D_esperado con el x local
de ese mismo corte, y salio un exceso de ~2.3. Pero x(m) no tiene meseta: sube con
pendiente ~0.7 por magnitud en esa zona. Asi que el exceso depende de donde cortes.

Este script recorre varios cortes de magnitud y recalcula LAS DOS COSAS de forma
consistente en cada uno:

    D_obs(m_c)  = dipolo medido con la muestra recortada en W1 < m_c
    D_esp(m_c)  = [2 + x(m_c)(1 + alpha(m_c))] * v/c

    cociente(m_c) = D_obs / D_esp

Si el cociente es plano -> el exceso es robusto frente a la seleccion.
Si deriva con el corte -> es un artefacto de seleccion, no una anomalia cosmologica.

Es la misma logica que ya te funciono con la mascara galactica: una cantidad fisica
real no debe depender de una eleccion arbitraria del analista.

NOTAS DE DISENO
  * x(m) se mide del PARCHE PROFUNDO (catwise_parche_profundo.ecsv), no del cielo
    completo. Razon: el limite de descarga del cielo completo trunca sus propios
    conteos, asi que su x local cerca del limite estaria sesgado. El parche llega a
    W1=18 y es completo hasta ~17.65, asi que da x limpio en todo el rango util.
  * alpha se mide como MEDIA del color (no mediana): D es lineal en alpha, asi que
    el valor correcto para una poblacion es el promedio.
  * El error de D_obs es analitico (exacto para este estimador lineal) y cubre solo
    el muestreo. NO incluye el clustering de cuasares, que es la contribucion que
    falta para una significancia publicable.

Requiere: numpy, matplotlib, astropy, pyvo (solo para la descarga).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# =============================== CONFIGURACION ===============================
ARCHIVO_CIELO  = "catwise_cielo_magnitudes.ecsv"   # cielo completo con w1mpro, w2mpro
ARCHIVO_PARCHE = "catwise_parche_profundo.ecsv"    # el que ya bajaste (deep patch)
CACHE_NPZ      = "cielo_cache.npz"                 # acelera ejecuciones posteriores

W1_LIMITE_DESCARGA = 17.0    # limite del cielo completo. 17.0 ~ 10M fuentes (~400 MB en ecsv).
                             # Bajalo a 16.8 si la descarga se hace pesada.
CORTES = np.arange(15.75, 16.80, 0.25)   # cortes de ciencia a probar
B_CUT  = 30.0                # corte galactico
K_ARTEFACTO = 12.0           # rechazo de celdas a media + K*sqrt(media) (Poisson)
N_RANDOM = 6_000_000

V_SOBRE_C = 369.82 / 299792.458
F0_W1, F0_W2   = 309.540, 171.787
LAM_W1, LAM_W2 = 3.3526, 4.6028
CMB_L, CMB_B   = 264.021, 48.253

# =============================== COORDENADAS ===============================
RA_NGP, DEC_NGP, L_NCP = np.radians(192.85948), np.radians(27.12825), np.radians(122.93192)

def eq2gal(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    b = np.arcsin(np.clip(np.sin(dec)*np.sin(DEC_NGP) +
                          np.cos(dec)*np.cos(DEC_NGP)*np.cos(ra-RA_NGP), -1, 1))
    y = np.cos(dec)*np.sin(ra-RA_NGP)
    x = np.sin(dec)*np.cos(DEC_NGP) - np.cos(dec)*np.sin(DEC_NGP)*np.cos(ra-RA_NGP)
    return np.degrees(L_NCP - np.arctan2(y, x)) % 360, np.degrees(b)

def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], axis=-1)

def sep_deg(l1, b1, l2, b2):
    return np.degrees(np.arccos(np.clip(np.sum(unit(l1, b1)*unit(l2, b2), axis=-1), -1, 1)))

# =============================== ESTIMADOR DE DIPOLO ===============================
def fit_dipolo(u_d, u_r):
    """dN/dOmega ~ (1 + D.n) sobre la region no enmascarada.
    <n>_datos = <n>_mask + M.D,  con M = <nn>_mask - <n>_mask<n>_mask.
    Devuelve amplitud, (l,b) y el error ANALITICO de la amplitud (solo muestreo).
    Nota: M se calcula como u.T@u/N, no con broadcasting 3D, para no reventar la RAM."""
    n_d, n_r = len(u_d), len(u_r)
    mu_d, mu_r = u_d.mean(0), u_r.mean(0)
    M = (u_r.T @ u_r) / n_r - np.outer(mu_r, mu_r)
    Minv = np.linalg.inv(M)
    D = Minv @ (mu_d - mu_r)
    amp = np.linalg.norm(D); nhat = D / amp
    # covarianza de mu_d -> covarianza de D -> error en la amplitud
    cov_mu = ((u_d.T @ u_d) / n_d - np.outer(mu_d, mu_d)) / n_d
    cov_D = Minv @ cov_mu @ Minv.T
    amp_err = np.sqrt(max(nhat @ cov_D @ nhat, 0.0))
    return (amp, amp_err,
            np.degrees(np.arctan2(nhat[1], nhat[0])) % 360,
            np.degrees(np.arcsin(np.clip(nhat[2], -1, 1))))

# =============================== x(m) DEL PARCHE ===============================
def pendiente_local(mags, m_grid, media_ventana=0.25):
    mags = np.sort(mags)
    out = np.full(len(m_grid), np.nan)
    for i, m0 in enumerate(m_grid):
        mm = np.linspace(m0 - media_ventana, m0 + media_ventana, 15)
        NN = np.searchsorted(mags, mm)
        g = NN > 30
        if g.sum() < 5:
            continue
        out[i] = np.polyfit(mm[g], np.log10(NN[g]), 1)[0] / 0.4
    return out

def color_a_alpha(c):
    return -np.log((F0_W1 / F0_W2) * 10 ** (-c / 2.5)) / np.log(LAM_W2 / LAM_W1)

# =============================== DESCARGA CIELO COMPLETO ===============================
def descargar_cielo():
    import pyvo as vo
    from astropy.table import Table, vstack
    service = vo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
    cuadrantes = [("Norte 1", "dec >= 0 AND ra < 180"), ("Norte 2", "dec >= 0 AND ra >= 180"),
                  ("Sur 1",   "dec < 0 AND ra < 180"),  ("Sur 2",   "dec < 0 AND ra >= 180")]
    tablas = []
    for nombre, cond in cuadrantes:
        q = f"""
        SELECT ra, dec, w1mpro, w2mpro
        FROM catwise_2020
        WHERE w1mpro - w2mpro >= 0.8
          AND w1snr >= 5 AND w2snr >= 5
          AND w1mpro < {W1_LIMITE_DESCARGA}
          AND {cond}
        """
        print(f"   [{nombre}] enviado a la cola de IRSA...")
        job = service.submit_job(q); job.execution_duration = 7200; job.run()
        job.wait(phases=['COMPLETED', 'ERROR', 'ABORTED'])
        if job.phase != 'COMPLETED':
            job.raise_if_error(); raise RuntimeError(job.phase)
        tablas.append(job.fetch_result().to_table())
        print(f"   [{nombre}] listo ({len(tablas[-1])} filas)")
    t = vstack(tablas)
    t.write(ARCHIVO_CIELO, overwrite=True)
    return t

print("1. Cielo completo con magnitudes")
if os.path.exists(CACHE_NPZ):
    z = np.load(CACHE_NPZ)
    ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
    print(f"   cache .npz cargado: {len(ra)} fuentes")
else:
    from astropy.table import Table
    if os.path.exists(ARCHIVO_CIELO):
        print(f"   leyendo {ARCHIVO_CIELO} (puede tardar, es grande)...")
        t = Table.read(ARCHIVO_CIELO)
    else:
        print(f"   descargando cielo completo hasta W1<{W1_LIMITE_DESCARGA}.")
        print("   Esto es lo pesado: varias decenas de minutos en la cola de NASA.")
        t = descargar_cielo()
    ra  = np.asarray(t['ra'], float);     dec = np.asarray(t['dec'], float)
    w1  = np.asarray(t['w1mpro'], float); w2  = np.asarray(t['w2mpro'], float)
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(w1) & np.isfinite(w2)
    ra, dec, w1, w2 = ra[ok], dec[ok], w1[ok], w2[ok]
    np.savez_compressed(CACHE_NPZ, ra=ra, dec=dec, w1=w1, w2=w2)
    print(f"   {len(ra)} fuentes  (cache .npz guardado para la proxima)")

gl, gb = eq2gal(ra, dec)

# =============================== x(m) ===============================
print("\n2. Midiendo x(m) en el parche profundo")
from astropy.table import Table
if not os.path.exists(ARCHIVO_PARCHE):
    raise SystemExit(f"Falta {ARCHIVO_PARCHE}. Corre primero medir_x_v2.py.")
tp = Table.read(ARCHIVO_PARCHE)
w1p = np.asarray(tp['w1mpro'], float)
w1p = w1p[np.isfinite(w1p)]
m_grid = np.arange(15.0, 17.5, 0.05)
x_grid = pendiente_local(w1p, m_grid)
print(f"   x(m) medido entre {m_grid[0]:.2f} y {m_grid[-1]:.2f}")
print(f"   (recuerda: el parche es completo hasta ~17.65; no uses cortes mas alla)")

# =============================== MASCARA Y RANDOMS ===============================
print("\n3. Construyendo mascara (galaxia + Nubes de Magallanes + celdas con artefactos)")
LMC, SMC, RAD = (80.894, -69.756), (13.187, -72.829), 8.0
rng = np.random.default_rng(1)
dr = np.degrees(np.arcsin(rng.uniform(-1, 1, N_RANDOM)))
rr = rng.uniform(0, 360, N_RANDOM)
glr, gbr = eq2gal(rr, dr)

base_d = (np.abs(gb) > B_CUT) & (sep_deg(ra, dec, *LMC) > RAD) & (sep_deg(ra, dec, *SMC) > RAD)
base_r = (np.abs(gbr) > B_CUT) & (sep_deg(rr, dr, *LMC) > RAD) & (sep_deg(rr, dr, *SMC) > RAD)

NB = 360
def celda(r, d):
    i = np.clip(((np.sin(np.radians(d)) + 1) / 2 * NB).astype(int), 0, NB - 1)
    j = np.clip((r / 360 * NB).astype(int), 0, NB - 1)
    return i * NB + j
cid_d, cid_r = celda(ra, dec), celda(rr, dr)

# =============================== BUCLE DE CORTES ===============================
print("\n4. TEST DE ESTABILIDAD FRENTE AL CORTE DE MAGNITUD")
print(f"   {'corte':>6} {'N':>9} {'x':>6} {'alpha':>6} {'D_esp':>8} {'D_obs':>17} "
      f"{'cociente':>14} {'sep':>6}")

res = []
for mc in CORTES:
    sel_d = base_d & (w1 < mc)
    # rechazo de artefactos, recalculado para esta muestra
    hb = np.bincount(cid_d[sel_d], minlength=NB * NB).astype(float)
    occ = hb[hb > 0]; mu_cell = occ.mean()
    malas = np.where(hb > mu_cell + K_ARTEFACTO * np.sqrt(mu_cell))[0]
    sel_d &= ~np.isin(cid_d, malas)
    sel_r = base_r & ~np.isin(cid_r, malas)      # la misma mascara en los randoms

    u_d = unit(gl[sel_d], gb[sel_d]); u_r = unit(glr[sel_r], gbr[sel_r])
    D_obs, D_err, lon, lat = fit_dipolo(u_d, u_r)

    x_c = np.interp(mc, m_grid, x_grid)
    alpha_c = color_a_alpha(np.mean((w1 - w2)[sel_d]))     # MEDIA, no mediana
    D_esp = (2 + x_c * (1 + alpha_c)) * V_SOBRE_C

    # incertidumbre sistematica en x por la pendiente de x(m): +-0.1 mag de corte
    dx = abs(np.interp(mc + 0.1, m_grid, x_grid) - np.interp(mc - 0.1, m_grid, x_grid)) / 2
    D_esp_err = V_SOBRE_C * (1 + alpha_c) * dx

    coc = D_obs / D_esp
    coc_err = coc * np.hypot(D_err / D_obs, D_esp_err / D_esp)
    s = sep_deg(lon, lat, CMB_L, CMB_B)
    print(f"   {mc:6.2f} {sel_d.sum():9d} {x_c:6.3f} {alpha_c:6.3f} {D_esp:8.5f} "
          f"{D_obs:.5f}+/-{D_err:.5f} {coc:6.2f}+/-{coc_err:4.2f} {s:6.1f}")
    res.append((mc, sel_d.sum(), x_c, alpha_c, D_esp, D_esp_err, D_obs, D_err, coc, coc_err, s))

res = np.array(res)
mc, x_, al, De, Dee, Do, Doe, co, coe, sp = (res[:, i] for i in
                                             (0, 2, 3, 4, 5, 6, 7, 8, 9, 10))

# chi2 contra un cociente constante: cuantifica si el cociente es plano
w = 1 / coe**2
co_med = np.sum(co * w) / np.sum(w)
chi2 = np.sum((co - co_med)**2 * w); dof = len(co) - 1
print(f"\n   cociente medio = {co_med:.2f}")
print(f"   chi2 contra cociente constante = {chi2:.1f} con {dof} g.l. "
      f"(chi2/dof = {chi2/dof:.2f})")
print("   chi2/dof ~ 1  -> plano, el exceso es robusto frente a la seleccion")
print("   chi2/dof >> 1 -> deriva con el corte: artefacto de seleccion")
print("\n   AVISO: el error de D_obs es solo de muestreo. El clustering de cuasares")
print("   anade varianza extra; para una significancia real hacen falta simulaciones.")

# =============================== FIGURA ===============================
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), dpi=140)

axs[0].plot(m_grid, x_grid, color='steelblue', lw=2)
axs[0].plot(mc, x_, 'o', color='crimson', ms=6)
axs[0].set_xlabel("corte W1"); axs[0].set_ylabel("x local")
axs[0].set_title("x(m): sin meseta = sin ley de potencias"); axs[0].grid(alpha=.3)

axs[1].errorbar(mc, Do * 100, yerr=Doe * 100, fmt='o-', color='crimson',
                lw=2, ms=6, capsize=3, label=r'$D_{obs}$')
axs[1].errorbar(mc, De * 100, yerr=Dee * 100, fmt='s--', color='0.35',
                lw=2, ms=5, capsize=3, label=r'$D_{esp}$ (Ellis-Baldwin)')
axs[1].set_xlabel("corte W1"); axs[1].set_ylabel("amplitud (%)")
axs[1].set_title("Observado vs esperado"); axs[1].legend(fontsize=9); axs[1].grid(alpha=.3)

axs[2].errorbar(mc, co, yerr=coe, fmt='o-', color='darkgreen', lw=2, ms=7, capsize=4)
axs[2].axhline(co_med, color='darkgreen', ls='--', lw=1.2, alpha=.7,
               label=f'media {co_med:.2f}')
axs[2].axhline(1.0, color='k', ls=':', lw=1.5, label='sin anomalia')
axs[2].axhspan(1.9, 2.1, color='steelblue', alpha=.18, label='Secrest+2021 (~2x)')
axs[2].set_xlabel("corte W1"); axs[2].set_ylabel(r"$D_{obs}/D_{esp}$")
axs[2].set_title(f"EL TEST: ¿es plano?  ($\\chi^2$/gl = {chi2/dof:.2f})")
axs[2].legend(fontsize=8); axs[2].grid(alpha=.3); axs[2].set_ylim(0, max(3.2, co.max() * 1.25))

plt.tight_layout()
plt.savefig("estabilidad_corte.png", dpi=140)
print("\nFigura guardada: estabilidad_corte.png")
plt.show()
