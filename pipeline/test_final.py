"""
test_final.py
-------------
Encontrar la mejor muestra de CatWISE combinando lo aprendido:

  * El sesgo de dispersion de color contamina el tramo pegado a W1-W2 = 0.8.
    Diagnosticado por cuatro vias: dependencia con el color, con la magnitud, con la
    cobertura, y el signo de esa dependencia (mas cobertura -> MENOS fuentes, que solo
    se explica por dispersion cruzando el umbral, no por profundidad de deteccion).
  * Los fragmentos de galaxias cercanas contaminan la cola roja y estan agrumados.
  * El rechazo de celdas a 12 sigma ya quitaba parte de eso.

La idea: entre el sesgo de color (abajo) y los fragmentos (arriba) hay una ventana
limpia. Si existe, permite usar un corte de magnitud mas profundo -y por tanto mucha
mas estadistica- manteniendo la direccion pegada al CMB.

=============================================================================
ADVERTENCIA METODOLOGICA, LEELA ANTES DE USAR EL RESULTADO
=============================================================================
Probar 32 configuraciones y quedarse con la que da el dipolo mas bonito es hacer
trampa: con suficientes cortes SIEMPRE aparece uno que da lo que quieres. Es el
problema del jardin de senderos que se bifurcan.

Por eso este script NO elige por el dipolo. Elige por el GRADIENTE ECLIPTICO, que
es una metrica de contaminacion completamente independiente del dipolo: mide si la
densidad de fuentes depende de la latitud ecliptica, cosa que ninguna fisica
cosmologica puede producir. Se fija un umbral de gradiente ACEPTABLE, se eligen las
configuraciones que lo cumplen, y RECIEN ENTONCES se mira el dipolo.

Elegir con una metrica y reportar otra es lo que hace que el resultado valga.
=============================================================================

Requiere: numpy, matplotlib, astropy. Usa cielo_cache.npz y catwise_parche_profundo.ecsv.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

CACHE   = "cielo_cache.npz"
PARCHE  = "catwise_parche_profundo.ecsv"
B_CUT, RAD_MC = 30.0, 8.0
K_ART   = 12.0
NB      = 360
N_RANDOM = 6_000_000

COLOR_MIN = [0.80, 0.90, 1.00, 1.10]
COLOR_MAX = [9.99, 1.60]              # 9.99 = sin corte superior
W1_MAX    = [16.00, 16.25, 16.50, 16.75]

GRADIENTE_ACEPTABLE = 0.03            # |densidad_plano/polo - 1| maximo admitido

V_SOBRE_C = 369.82/299792.458
CMB_L, CMB_B = 264.021, 48.253
LMC, SMC = (80.894, -69.756), (13.187, -72.829)
F0_W1, F0_W2, LAM_W1, LAM_W2 = 309.540, 171.787, 3.3526, 4.6028

# =============================== COORDENADAS ===============================
RA_NGP, DEC_NGP, L_NCP = np.radians(192.85948), np.radians(27.12825), np.radians(122.93192)
EPS = np.radians(23.4392911)

def eq2gal(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    b = np.arcsin(np.clip(np.sin(dec)*np.sin(DEC_NGP) +
                          np.cos(dec)*np.cos(DEC_NGP)*np.cos(ra-RA_NGP), -1, 1))
    y = np.cos(dec)*np.sin(ra-RA_NGP)
    x = np.sin(dec)*np.cos(DEC_NGP) - np.cos(dec)*np.sin(DEC_NGP)*np.cos(ra-RA_NGP)
    return np.degrees(L_NCP - np.arctan2(y, x)) % 360, np.degrees(b)

def eq2ecl(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    beta = np.arcsin(np.clip(np.sin(dec)*np.cos(EPS) - np.cos(dec)*np.sin(EPS)*np.sin(ra), -1, 1))
    return np.degrees(beta)

def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], axis=-1)

def vec2lb(v):
    n = v/np.linalg.norm(v)
    return np.degrees(np.arctan2(n[1], n[0])) % 360, np.degrees(np.arcsin(np.clip(n[2], -1, 1)))

def sep_deg(l1, b1, l2, b2):
    return np.degrees(np.arccos(np.clip(np.sum(unit(l1, b1)*unit(l2, b2), axis=-1), -1, 1)))

def color_a_alpha(c):
    return -np.log((F0_W1/F0_W2)*10**(-c/2.5))/np.log(LAM_W2/LAM_W1)

def pendiente_local(mags, m0, ventana=0.30):
    mags = np.sort(mags[np.isfinite(mags)])
    mm = np.linspace(m0-ventana, m0+ventana, 20)
    NN = np.searchsorted(mags, mm); g = NN > 30
    return np.polyfit(mm[g], np.log10(NN[g]), 1)[0]/0.4 if g.sum() >= 5 else np.nan

# =============================== CARGA ===============================
print("1. Carga")
z = np.load(CACHE)
ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
gl, gb = eq2gal(ra, dec); eb = eq2ecl(ra, dec); color = w1 - w2
print(f"   {len(ra)} fuentes")

from astropy.table import Table
tp = Table.read(PARCHE)
cp = {c.lower(): c for c in tp.colnames}
w1p = np.asarray(tp[cp['w1mpro']], float)
w2p = np.asarray(tp[cp['w2mpro']], float) if 'w2mpro' in cp else None
colp = w1p - w2p if w2p is not None else None
print(f"   parche: {len(w1p)} fuentes" + ("" if colp is not None else "  (sin w2mpro: x no se ajusta por color)"))

# =============================== MASCARA ===============================
print("\n2. Mascara y rechazo de artefactos")
rng = np.random.default_rng(1)
dr = np.degrees(np.arcsin(rng.uniform(-1, 1, N_RANDOM))); rr = rng.uniform(0, 360, N_RANDOM)
glr, gbr = eq2gal(rr, dr); ebr = eq2ecl(rr, dr)
base_d = (np.abs(gb) > B_CUT) & (sep_deg(ra, dec, *LMC) > RAD_MC) & (sep_deg(ra, dec, *SMC) > RAD_MC)
base_r = (np.abs(gbr) > B_CUT) & (sep_deg(rr, dr, *LMC) > RAD_MC) & (sep_deg(rr, dr, *SMC) > RAD_MC)

def celda(r, d):
    i = np.clip(((np.sin(np.radians(d))+1)/2*NB).astype(int), 0, NB-1)
    j = np.clip((r/360*NB).astype(int), 0, NB-1)
    return i*NB + j
cid_d, cid_r = celda(ra, dec), celda(rr, dr)
prof = base_d & (w1 < max(W1_MAX))
hb = np.bincount(cid_d[prof], minlength=NB*NB).astype(float)
occ = hb[hb > 0]; mu_c = occ.mean()
malas = np.where(hb > mu_c + K_ART*np.sqrt(mu_c))[0]
en_mala_d, en_mala_r = np.isin(cid_d, malas), np.isin(cid_r, malas)
print(f"   {len(malas)} celdas rechazadas, {en_mala_d.sum()} fuentes ({100*en_mala_d.mean():.2f}%)")

# --- el test de 30 segundos: ¿que color tienen las fuentes de las celdas rechazadas? ---
c_mal = color[prof & en_mala_d]; c_ok = color[prof & ~en_mala_d]
print(f"\n   COLOR DENTRO DE LAS CELDAS RECHAZADAS vs FUERA")
print(f"      dentro : mediana={np.median(c_mal):.3f}  "
      f"fraccion con W1-W2>1.2 = {100*(c_mal > 1.2).mean():.1f}%")
print(f"      fuera  : mediana={np.median(c_ok):.3f}  "
      f"fraccion con W1-W2>1.2 = {100*(c_ok > 1.2).mean():.1f}%")
print("      (si la de dentro es MUCHO mas roja, los fragmentos y las celdas son lo mismo)")

u_r_full = unit(glr[base_r & ~en_mala_r], gbr[base_r & ~en_mala_r])

# =============================== ESTIMADOR ===============================
def fit(u_d, u_r):
    n = len(u_d); mu_d = u_d.mean(0)
    cov_mu = ((u_d.T @ u_d)/n - np.outer(mu_d, mu_d))/n
    mu_r = u_r.mean(0); M = (u_r.T @ u_r)/len(u_r) - np.outer(mu_r, mu_r)
    Minv = np.linalg.inv(M); D = Minv @ (mu_d - mu_r)
    covD = Minv @ cov_mu @ Minv.T
    a = np.linalg.norm(D); nh = D/a
    return D, a, np.sqrt(max(nh @ covD @ nh, 0.0))

bins_b = np.linspace(-90, 90, 19); cen_b = 0.5*(bins_b[1:]+bins_b[:-1])
area_r, _ = np.histogram(ebr[base_r & ~en_mala_r], bins=bins_b)
plano_m, polo_m = np.abs(cen_b) < 25, np.abs(cen_b) > 60

def gradiente_ecl(mask):
    h, _ = np.histogram(eb[mask], bins=bins_b)
    d = h/np.maximum(area_r, 1)
    return (np.average(d[plano_m], weights=area_r[plano_m]) /
            np.average(d[polo_m], weights=area_r[polo_m]))

n_cmb = unit(CMB_L, CMB_B)

# =============================== DIAGNOSTICO POR COLOR ===============================
print("\n3. TRAMOS DE COLOR, AHORA CON EL RECHAZO DE CELDAS ACTIVO")
print(f"   {'color':>12} {'N':>9} {'grad ecl':>9} {'|D|':>17} {'sep CMB':>8} {'D_perp':>8}")
for c0, c1 in [(0.8, 0.9), (0.9, 1.0), (1.0, 1.2), (1.2, 1.6), (1.6, 9.99)]:
    m = base_d & ~en_mala_d & (color >= c0) & (color < c1) & (w1 < 16.5)
    if m.sum() < 20000:
        continue
    D, a, ae = fit(unit(gl[m], gb[m]), u_r_full); l, b = vec2lb(D)
    perp = np.linalg.norm(D - (D @ n_cmb)*n_cmb)
    print(f"   {c0:5.2f}-{c1:5.2f} {m.sum():9d} {gradiente_ecl(m):9.3f} "
          f"{a:.5f}+/-{ae:.5f} {sep_deg(l,b,CMB_L,CMB_B):8.1f} {perp:8.5f}")

# =============================== REJILLA ===============================
print("\n" + "=" * 92)
print("4. REJILLA COMPLETA")
print(f"   La seleccion se hace por |grad-1| < {GRADIENTE_ACEPTABLE}, NO por el dipolo.\n")
print(f"   {'cmin':>5} {'cmax':>5} {'W1<':>6} {'N':>9} {'grad':>7} {'x':>6} {'alpha':>6} "
      f"{'D_esp':>8} {'|D|':>9} {'sep':>6} {'perp/sig':>9} {'coc':>12} {'OK':>4}")
res = []
for cmax in COLOR_MAX:
    for cmin in COLOR_MIN:
        for wm in W1_MAX:
            m = base_d & ~en_mala_d & (color >= cmin) & (color < cmax) & (w1 < wm)
            if m.sum() < 50000:
                continue
            g = gradiente_ecl(m)
            D, a, ae = fit(unit(gl[m], gb[m]), u_r_full)
            l, b = vec2lb(D)
            par = D @ n_cmb; perp = np.linalg.norm(D - par*n_cmb)
            if colp is not None:
                mp = (colp >= cmin) & (colp < cmax)
                xv = pendiente_local(w1p[mp], wm)
            else:
                xv = pendiente_local(w1p, wm)
            al = color_a_alpha(np.mean(color[m]))
            De = (2 + xv*(1+al))*V_SOBRE_C
            ok = abs(g - 1) < GRADIENTE_ACEPTABLE
            print(f"   {cmin:5.2f} {cmax:5.2f} {wm:6.2f} {m.sum():9d} {g:7.3f} {xv:6.3f} "
                  f"{al:6.3f} {De:8.5f} {a:.5f} {sep_deg(l,b,CMB_L,CMB_B):6.1f} "
                  f"{perp/ae:9.1f} {a/De:5.2f}+/-{ae/De:4.2f} {'SI' if ok else 'no':>4}")
            res.append((cmin, cmax, wm, m.sum(), g, xv, al, De, a, ae,
                        sep_deg(l, b, CMB_L, CMB_B), par, perp, perp/ae, a/De, ae/De))
res = np.array(res)

# =============================== SELECCION ===============================
print("\n" + "=" * 92)
print("5. SELECCION  (filtro por gradiente; entre las que pasan, la mas precisa)")
buenas = res[np.abs(res[:, 4]-1) < GRADIENTE_ACEPTABLE]
if len(buenas) == 0:
    print("   Ninguna configuracion pasa el filtro. Relaja GRADIENTE_ACEPTABLE.")
else:
    orden = buenas[np.argsort(buenas[:, 15])]            # menor error en el cociente
    print(f"   {len(buenas)} de {len(res)} configuraciones pasan.\n")
    print(f"   {'cmin':>5} {'cmax':>5} {'W1<':>6} {'N':>9} {'grad':>7} {'sep':>6} "
          f"{'perp/sig':>9} {'cociente':>13}")
    for r in orden[:6]:
        print(f"   {r[0]:5.2f} {r[1]:5.2f} {r[2]:6.2f} {int(r[3]):9d} {r[4]:7.3f} "
              f"{r[10]:6.1f} {r[13]:9.1f} {r[14]:6.2f}+/-{r[15]:5.2f}")
    b = orden[0]
    print(f"\n   >>> MEJOR MUESTRA: W1-W2 en [{b[0]:.2f}, {b[1]:.2f}), W1 < {b[2]:.2f}")
    print(f"       N = {int(b[3])}   gradiente ecliptico = {b[4]:.3f}")
    print(f"       |D| = {b[8]:.5f} +/- {b[9]:.5f}   a {b[10]:.1f} grados del CMB")
    print(f"       componente perpendicular = {b[12]:.5f}  ({b[13]:.1f} sigma)")
    print(f"       cociente = {b[14]:.2f} +/- {b[15]:.2f}")

    # comparacion con la mejor muestra anterior
    m0 = base_d & ~en_mala_d & (color >= 0.8) & (w1 < 16.00)
    D0, a0, ae0 = fit(unit(gl[m0], gb[m0]), u_r_full); l0, b0_ = vec2lb(D0)
    p0 = np.linalg.norm(D0 - (D0 @ n_cmb)*n_cmb)
    print(f"\n   Comparacion con la mejor muestra anterior (color>=0.8, W1<16.00):")
    print(f"       N = {m0.sum()}   |D| = {a0:.5f} +/- {ae0:.5f}   "
          f"sep = {sep_deg(l0,b0_,CMB_L,CMB_B):.1f}   perp = {p0:.5f} ({p0/ae0:.1f} sigma)")
    print(f"       ganancia en fuentes: x{b[3]/m0.sum():.2f}   "
          f"cambio en el error del cociente: {100*(b[15]/(ae0/b[7])-1):+.0f}%")

# =============================== FIGURA ===============================
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=145)
sin_top = res[res[:, 1] > 9]
col = plt.cm.viridis(np.linspace(0, .85, len(COLOR_MIN)))
for i, cmin in enumerate(COLOR_MIN):
    s = sin_top[sin_top[:, 0] == cmin]
    if len(s) == 0:
        continue
    axs[0].plot(s[:, 2], s[:, 4], '-o', color=col[i], lw=2, ms=6, label=f"W1-W2 ≥ {cmin:.2f}")
    axs[1].plot(s[:, 2], s[:, 10], '-o', color=col[i], lw=2, ms=6)
    axs[2].errorbar(s[:, 2], s[:, 14], yerr=s[:, 15], fmt='-o', color=col[i], lw=2, ms=6, capsize=3)
axs[0].axhspan(1-GRADIENTE_ACEPTABLE, 1+GRADIENTE_ACEPTABLE, color='green', alpha=.12,
               label='zona aceptable')
axs[0].axhline(1, color='k', ls=':', lw=1.2)
axs[0].set_xlabel("corte W1"); axs[0].set_ylabel("densidad plano / polo eclíptico")
axs[0].set_title("MÉTRICA DE SELECCIÓN (independiente del dipolo)")
axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
axs[1].axhline(0, color='k', ls=':', lw=1.5)
axs[1].set_xlabel("corte W1"); axs[1].set_ylabel("separación del CMB (grados)")
axs[1].set_title("Dirección"); axs[1].grid(alpha=.3)
axs[2].axhline(1, color='k', ls=':', lw=1.2)
axs[2].set_xlabel("corte W1"); axs[2].set_ylabel(r"$D_{obs}/D_{esp}$")
axs[2].set_title("Cociente"); axs[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig("test_final.png", dpi=145)
print("\nFigura guardada: test_final.png")
plt.show()

print("""
COMO USAR ESTO HONESTAMENTE

 * El corte se justifica por el MECANISMO que identificaste (dispersion de color cerca
   del umbral, fragmentos en la cola roja), no por el dipolo que produce. Si alguien te
   pregunta por que cortaste donde cortaste, la respuesta es el gradiente ecliptico y
   el test diferencial de cobertura, no "porque asi salia mejor".
 * Si la configuracion elegida por el gradiente resulta NO ser la que da el dipolo mas
   bonito, reportala igual. Eso es precisamente lo que hace creible al resultado.
 * Las barras siguen sin incluir clustering. El cociente que salga de aqui es una
   medicion, no una significancia.
""")
