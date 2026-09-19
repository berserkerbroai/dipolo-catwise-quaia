"""
contaminante.py
---------------
Tres cosas que el analisis anterior dejo abiertas:

  A) El chi2 = 0.22 no vale, porque las cinco muestras estan ANIDADAS y por tanto
     correlacionadas. Aqui se hace un BOOTSTRAP CONJUNTO: se remuestrea el catalogo
     UNA vez y se recalculan los cinco cortes desde ese mismo remuestreo. Eso da la
     matriz de covarianza entre cortes y con ella un chi2 legitimo.

  B) Se imprime (l, b) de cada ajuste, no solo la separacion del CMB.

  C) Se aisla el CONTAMINANTE. Dos vias:
       - la componente del dipolo perpendicular a la direccion del CMB, corte a corte
       - el analisis por CAPAS de magnitud (shells), que son muestras DISJUNTAS y por
         tanto estadisticamente independientes: la capa [16.50, 16.75) contiene solo
         las fuentes que se anaden al profundizar, asi que su dipolo expone al
         contaminante casi sin diluir.
     Luego se compara esa direccion contra una lista de sospechosos (plano galactico,
     polos eclipticos, Nubes de Magallanes, etc.) para ponerle nombre.

NOTA SOBRE LA MASCARA DE ARTEFACTOS: aqui se calcula UNA sola vez, con la muestra mas
profunda, y se aplica igual a todos los cortes. Es imprescindible para que los cortes
sean exactamente anidados y la covarianza del bootstrap signifique algo. (En
estabilidad_corte.py se recalculaba por corte, lo cual es mas fino pero rompe el
anidamiento exacto.)

Requiere: numpy, matplotlib, astropy. Usa cielo_cache.npz y catwise_parche_profundo.ecsv.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# =============================== CONFIGURACION ===============================
CACHE_NPZ      = "cielo_cache.npz"
ARCHIVO_PARCHE = "catwise_parche_profundo.ecsv"
CORTES   = np.array([15.75, 16.00, 16.25, 16.50, 16.75])
B_CUT    = 30.0
K_ARTEFACTO = 12.0
N_RANDOM = 6_000_000
N_BOOT   = 200

V_SOBRE_C = 369.82 / 299792.458
F0_W1, F0_W2   = 309.540, 171.787
LAM_W1, LAM_W2 = 3.3526, 4.6028
CMB_L, CMB_B   = 264.021, 48.253

# =============================== COORDENADAS ===============================
RA_NGP, DEC_NGP, L_NCP = np.radians(192.85948), np.radians(27.12825), np.radians(122.93192)
EPS = np.radians(23.4392911)

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

def vec2lb(v):
    n = v / np.linalg.norm(v)
    return np.degrees(np.arctan2(n[1], n[0])) % 360, np.degrees(np.arcsin(np.clip(n[2], -1, 1)))

def sep_deg(l1, b1, l2, b2):
    return np.degrees(np.arccos(np.clip(np.sum(unit(l1, b1)*unit(l2, b2), axis=-1), -1, 1)))

def color_a_alpha(c):
    return -np.log((F0_W1 / F0_W2) * 10 ** (-c / 2.5)) / np.log(LAM_W2 / LAM_W1)

def pendiente_local(mags, m_grid, media_ventana=0.25):
    mags = np.sort(mags)
    out = np.full(len(m_grid), np.nan)
    for i, m0 in enumerate(m_grid):
        mm = np.linspace(m0 - media_ventana, m0 + media_ventana, 15)
        NN = np.searchsorted(mags, mm); g = NN > 30
        if g.sum() >= 5:
            out[i] = np.polyfit(mm[g], np.log10(NN[g]), 1)[0] / 0.4
    return out

# =============================== CARGA ===============================
print("1. Cargando datos")
if not os.path.exists(CACHE_NPZ):
    raise SystemExit(f"Falta {CACHE_NPZ}. Corre primero estabilidad_corte.py.")
z = np.load(CACHE_NPZ)
ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
gl, gb = eq2gal(ra, dec)
print(f"   {len(ra)} fuentes")

from astropy.table import Table
tp = Table.read(ARCHIVO_PARCHE)
w1p = np.asarray(tp['w1mpro'], float); w1p = w1p[np.isfinite(w1p)]
m_grid = np.arange(15.0, 17.5, 0.05)
x_grid = pendiente_local(w1p, m_grid)

# =============================== MASCARA (FIJA) ===============================
print("2. Mascara: galaxia, Nubes de Magallanes, celdas con artefactos (calculada UNA vez)")
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

prof = base_d & (w1 < CORTES[-1])
hb = np.bincount(cid_d[prof], minlength=NB*NB).astype(float)
occ = hb[hb > 0]; mu_cell = occ.mean()
malas = np.where(hb > mu_cell + K_ARTEFACTO*np.sqrt(mu_cell))[0]
print(f"   {len(malas)} celdas rechazadas (umbral {mu_cell:.1f} + {K_ARTEFACTO}*sqrt)")

sel_par = prof & ~np.isin(cid_d, malas)          # muestra padre (corte mas profundo)
sel_r   = base_r & ~np.isin(cid_r, malas)
u_r = unit(glr[sel_r], gbr[sel_r])
mu_r = u_r.mean(0)
M = (u_r.T @ u_r)/len(u_r) - np.outer(mu_r, mu_r)
Minv = np.linalg.inv(M)
print(f"   muestra padre: {sel_par.sum()} fuentes ; randoms utiles: {len(u_r)}")

# Cada fuente pertenece a los cortes j >= jmin
u_par  = unit(gl[sel_par], gb[sel_par])
w1_par = w1[sel_par]
col_par = (w1 - w2)[sel_par]
jmin = np.searchsorted(CORTES, w1_par, side='left')
n_par = len(u_par)

def dipolos_desde_pesos(pesos):
    """Dado un vector de multiplicidades por fuente, devuelve los 5 vectores dipolo
    (acumulados = cortes) y los 5 de capa (disjuntos = shells)."""
    S = np.stack([np.bincount(jmin, weights=pesos*u_par[:, d], minlength=len(CORTES))
                  for d in range(3)], axis=1)              # (5,3) suma por capa
    N = np.bincount(jmin, weights=pesos, minlength=len(CORTES))
    Sc, Nc = np.cumsum(S, axis=0), np.cumsum(N)            # acumulados = cortes
    D_cut   = np.array([Minv @ (Sc[k]/Nc[k] - mu_r) for k in range(len(CORTES))])
    D_shell = np.array([Minv @ (S[k]/N[k] - mu_r) if N[k] > 0 else np.zeros(3)
                        for k in range(len(CORTES))])
    return D_cut, D_shell, Nc, N

# =============================== MEDICION CENTRAL ===============================
print("\n3. DIPOLO POR CORTE (acumulado)")
pesos1 = np.ones(n_par)
D_cut, D_shell, Nc, Nsh = dipolos_desde_pesos(pesos1)
n_cmb = unit(CMB_L, CMB_B)

x_c   = np.interp(CORTES, m_grid, x_grid)
alpha = np.array([color_a_alpha(col_par[jmin <= k].mean()) for k in range(len(CORTES))])
D_esp = (2 + x_c*(1 + alpha))*V_SOBRE_C

print(f"   {'corte':>6} {'N':>9} {'|D|':>8} {'l':>7} {'b':>7} {'sep':>6} "
      f"{'D_par':>8} {'D_perp':>8} {'coc':>6}")
par = D_cut @ n_cmb
for k, mc in enumerate(CORTES):
    l, b = vec2lb(D_cut[k]); amp = np.linalg.norm(D_cut[k])
    perp = np.linalg.norm(D_cut[k] - par[k]*n_cmb)
    print(f"   {mc:6.2f} {int(Nc[k]):9d} {amp:8.5f} {l:7.1f} {b:+7.1f} "
          f"{sep_deg(l,b,CMB_L,CMB_B):6.1f} {par[k]:8.5f} {perp:8.5f} {amp/D_esp[k]:6.2f}")

# =============================== BOOTSTRAP CONJUNTO ===============================
print(f"\n4. BOOTSTRAP CONJUNTO ({N_BOOT} remuestreos del MISMO catalogo)")
boot_r = np.empty((N_BOOT, len(CORTES)))
boot_perp = np.empty((N_BOOT, len(CORTES)))
for i in range(N_BOOT):
    pesos = np.bincount(rng.integers(0, n_par, n_par), minlength=n_par).astype(float)
    Dc, Ds, _, _ = dipolos_desde_pesos(pesos)
    boot_r[i] = np.linalg.norm(Dc, axis=1)/D_esp
    p = Dc @ n_cmb
    boot_perp[i] = np.linalg.norm(Dc - p[:, None]*n_cmb[None, :], axis=1)

C = np.cov(boot_r.T)
sig = np.sqrt(np.diag(C))
corr = C/np.outer(sig, sig)
print("   errores del cociente:", " ".join(f"{s:.2f}" for s in sig))
print("   matriz de CORRELACION entre cortes (por eso el chi2 ingenuo mentia):")
for fila in corr:
    print("      " + "  ".join(f"{v:+.2f}" for v in fila))

r_obs = np.linalg.norm(D_cut, axis=1)/D_esp
Cinv = np.linalg.pinv(C)
uno = np.ones(len(CORTES))
r_med = (uno @ Cinv @ r_obs)/(uno @ Cinv @ uno)
chi2 = (r_obs - r_med) @ Cinv @ (r_obs - r_med)
dof = len(CORTES) - 1
print(f"\n   cociente medio (pesado con covarianza) = {r_med:.2f}")
print(f"   chi2 CON covarianza = {chi2:.2f} / {dof} g.l.  (chi2/dof = {chi2/dof:.2f})")
print(f"   [el chi2 ingenuo, sin covarianza, daba 0.22 y no era interpretable]")

# =============================== CAPAS DISJUNTAS ===============================
print("\n5. CAPAS DE MAGNITUD (disjuntas => estadisticamente INDEPENDIENTES)")
bordes = ["<%.2f" % CORTES[0]] + ["%.2f-%.2f" % (CORTES[k-1], CORTES[k])
                                  for k in range(1, len(CORTES))]
print(f"   {'capa':>13} {'N':>9} {'|D|':>8} {'l':>7} {'b':>7} {'sep CMB':>8}")
for k in range(len(CORTES)):
    if Nsh[k] < 1000:
        continue
    l, b = vec2lb(D_shell[k])
    print(f"   {bordes[k]:>13} {int(Nsh[k]):9d} {np.linalg.norm(D_shell[k]):8.5f} "
          f"{l:7.1f} {b:+7.1f} {sep_deg(l,b,CMB_L,CMB_B):8.1f}")

# =============================== IDENTIFICAR AL CULPABLE ===============================
print("\n6. ¿HACIA DONDE APUNTA EL CONTAMINANTE?")
# residual = dipolo del corte profundo menos el del corte mas limpio
D_res = D_cut[-1] - D_cut[0]
l_res, b_res = vec2lb(D_res)
# componente perpendicular al CMB en el corte profundo
D_perp_vec = D_cut[-1] - par[-1]*n_cmb
l_perp, b_perp = vec2lb(D_perp_vec)

nep_l, nep_b = eq2gal(270.0, 90.0 - np.degrees(EPS))
sep_l, sep_b = eq2gal(90.0, -(90.0 - np.degrees(EPS)))
lmc_l, lmc_b = eq2gal(*LMC)
smc_l, smc_b = eq2gal(*SMC)
sospechosos = {
    "dipolo CMB":            (CMB_L, CMB_B),
    "centro galactico":      (0.0, 0.0),
    "anticentro galactico":  (180.0, 0.0),
    "polo norte galactico":  (0.0, 90.0),
    "polo sur galactico":    (0.0, -90.0),
    "polo ecliptico NORTE":  (nep_l, nep_b),
    "polo ecliptico SUR":    (sep_l, sep_b),
    "LMC":                   (lmc_l, lmc_b),
    "SMC":                   (smc_l, smc_b),
}
for etiqueta, (lv, bv) in [("residual (profundo - brillante)", (l_res, b_res)),
                           ("componente perpendicular al CMB", (l_perp, b_perp))]:
    print(f"\n   {etiqueta}: (l,b) = ({lv:.1f}, {bv:+.1f})")
    orden = sorted(sospechosos.items(), key=lambda kv: sep_deg(lv, bv, *kv[1]))
    for nombre, (cl, cb) in orden[:4]:
        print(f"      a {sep_deg(lv, bv, cl, cb):5.1f} grados de {nombre}")
print("\n   Un eje (p.ej. polos eclipticos o galacticos) es ambiguo en signo:")
print("   mira tambien la separacion al punto OPUESTO antes de concluir.")

# =============================== FIGURA ===============================
fig = plt.figure(figsize=(14, 5.6), dpi=145)
ax = fig.add_subplot(121, projection='mollweide')
def moll(l, b):
    return np.radians(((l + 180) % 360) - 180), np.radians(b)
for nombre, (cl, cb) in sospechosos.items():
    xx, yy = moll(cl, cb)
    ax.plot(xx, yy, 'x', color='0.55', ms=7, mew=1.6)
    ax.annotate(nombre, moll(cl, cb), fontsize=6.5, color='0.4',
                xytext=(4, 4), textcoords='offset points')
cm = plt.cm.viridis(np.linspace(0, .85, len(CORTES)))
for k, mc in enumerate(CORTES):
    l, b = vec2lb(D_cut[k]); xx, yy = moll(l, b)
    ax.plot(xx, yy, 'o', color=cm[k], ms=9, label=f"W1<{mc:.2f}")
xx, yy = moll(CMB_L, CMB_B)
ax.plot(xx, yy, '*', color='crimson', ms=20, label='dipolo CMB')
xx, yy = moll(l_perp, b_perp)
ax.plot(xx, yy, 'P', color='darkorange', ms=13, label='contaminante ⊥')
ax.grid(alpha=.3); ax.legend(fontsize=7, loc='lower left', bbox_to_anchor=(-0.16, -0.14))
ax.set_title("Direcciones en coordenadas galácticas", fontsize=10.5)

ax2 = fig.add_subplot(122)
im = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax2.set_xticks(range(len(CORTES))); ax2.set_yticks(range(len(CORTES)))
etiq = [f"{c:.2f}" for c in CORTES]
ax2.set_xticklabels(etiq); ax2.set_yticklabels(etiq)
for i in range(len(CORTES)):
    for j in range(len(CORTES)):
        ax2.text(j, i, f"{corr[i,j]:.2f}", ha='center', va='center', fontsize=8.5,
                 color='white' if abs(corr[i, j]) > .6 else 'black')
ax2.set_title(f"Correlación entre cortes anidados\n$\\chi^2$/gl con covarianza = {chi2/dof:.2f}",
              fontsize=10.5)
plt.colorbar(im, ax=ax2, fraction=.046)
plt.tight_layout(); plt.savefig("contaminante.png", dpi=145)
print("\nFigura guardada: contaminante.png")
plt.show()
