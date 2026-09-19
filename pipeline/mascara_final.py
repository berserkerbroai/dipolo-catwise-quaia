"""
mascara_final.py
----------------
Zanjar la discrepancia de mascara y producir el resultado final con presupuesto de
error completo.

EL PROBLEMA
  El mismo catalogo da |D| = 0.01389 con una definicion de mascara y 0.01504 con
  otra. 8% de diferencia.

    MASCARA A (continua):  la region |b|>30 se traza con un catalogo aleatorio (o una
                           rejilla fina). Las fuentes se filtran por SU PROPIA
                           latitud galactica. La mascara es la region real.
    MASCARA B (escalera):  la region se traza con centros de pixel. Las fuentes se
                           seleccionan por pertenencia a pixel. La mascara es una
                           aproximacion dentada de la region real.

  Ambas son internamente consistentes. Ninguna es obviamente "la correcta": para un
  dipolo puro las dos serian insesgadas. Que difieran en un 8% significa que a una de
  las dos se le esta filtrando potencia de multipolos altos a traves de la geometria
  de su borde — y la escalera tiene mucha mas potencia de borde que la region lisa.

LO QUE HACE ESTE SCRIPT
  No argumenta. MIDE. Genera simulaciones con un dipolo de amplitud CONOCIDA, las
  pasa por las dos mascaras, y compara lo recuperado con lo inyectado. La que
  devuelve el valor inyectado es la fiducial; la otra queda como sistematico.

  Despues produce el resultado final con los tres terminos del error:
    estadistico (clustering + Poisson, por simulaciones)
    sistematico de mascara (diferencia entre A y B)
    sistematico de la expectativa (incertidumbre en x y alpha)

Requiere: numpy, matplotlib, healpy.
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

# =============================== CONFIGURACION ===============================
CACHE = "cielo_cache.npz"
NSIDE_ANA  = 128      # resolucion del analisis por pixeles (mascara B)
NSIDE_FINA = 512      # rejilla fina que representa la region continua (mascara A)
NSIDE_MOCK = 256      # resolucion de las simulaciones

B_CUT, RAD_MC = 30.0, 8.0
K_ART, FWHM_SUAVE = 8.0, 5.0
COLOR_MIN, COLOR_MAX, W1_MAX = 0.80, 1.60, 16.00
W1_PROFUNDO = 16.75

D_ESPERADO     = 0.00670
D_ESPERADO_SYS = 0.00067     # 10%: refleja que x(m) no tiene meseta y alpha es incierto
N_MOCKS_BIAS   = 300
N_MOCKS_SIG    = 2000

CMB_L, CMB_B = 264.021, 48.253
LMC, SMC = (80.894, -69.756), (13.187, -72.829)
RA_NGP, DEC_NGP, L_NCP = np.radians(192.85948), np.radians(27.12825), np.radians(122.93192)

# =============================== UTILIDADES ===============================
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
    lmc_l, lmc_b = eq2gal(*LMC); smc_l, smc_b = eq2gal(*SMC)
    return ((np.abs(lat) > B_CUT)
            & (sep_deg(lon, lat, lmc_l, lmc_b) > RAD_MC)
            & (sep_deg(lon, lat, smc_l, smc_b) > RAD_MC))

def centros(nside):
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return np.degrees(ph), 90.0 - np.degrees(th)

def estimador(u_dat, pesos_dat, u_mask):
    """D = M^-1 (<n>_datos - <n>_mask). u_mask define la region."""
    mu_d = ((pesos_dat[:, None]*u_dat).sum(0)/pesos_dat.sum() if pesos_dat is not None
            else u_dat.mean(0))
    mu_r = u_mask.mean(0)
    M = (u_mask.T @ u_mask)/len(u_mask) - np.outer(mu_r, mu_r)
    return np.linalg.solve(M, mu_d - mu_r)

# =============================== 1. DATOS Y MASCARAS ===============================
print("=" * 78)
print("1. DATOS Y LAS DOS MASCARAS")
z = np.load(CACHE)
ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
gl, gb = eq2gal(ra, dec); color = w1 - w2
cien = (color >= COLOR_MIN) & (color < COLOR_MAX) & (w1 < W1_MAX)

# --- rechazo de artefactos con referencia local, en la rejilla de analisis ---
NP_A = hp.nside2npix(NSIDE_ANA)
pix_a = hp.ang2pix(NSIDE_ANA, np.radians(90.0 - gb), np.radians(gl))
l_a, b_a = centros(NSIDE_ANA)
geom_a = geom_de(l_a, b_a)
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
malas_a = geom_a & (exc > K_ART)
print(f"   {malas_a.sum()} pixeles rechazados por artefactos (referencia local {FWHM_SUAVE} grados)")

# --- MASCARA B: escalera de centros de pixel, a NSIDE_ANA ---
maskB = geom_a & ~malas_a
uB = unit(l_a[maskB], b_a[maskB])
selB = cien & maskB[pix_a]

# --- MASCARA A: region continua, representada por una rejilla fina ---
l_f, b_f = centros(NSIDE_FINA)
pix_f_a = hp.ang2pix(NSIDE_ANA, np.radians(90.0 - b_f), np.radians(l_f))
maskA_fina = geom_de(l_f, b_f) & ~malas_a[pix_f_a]      # mismos artefactos, borde liso
uA = unit(l_f[maskA_fina], b_f[maskA_fina])
selA = cien & geom_de(gl, gb) & ~malas_a[pix_a]

print(f"   mascara A (continua, rejilla NSIDE={NSIDE_FINA}): f_sky = {maskA_fina.mean():.4f}, "
      f"{selA.sum()} fuentes")
print(f"   mascara B (escalera,  NSIDE={NSIDE_ANA}):        f_sky = {maskB.mean():.4f}, "
      f"{selB.sum()} fuentes")

# =============================== 2. DIPOLO CON CADA UNA ===============================
print("\n" + "=" * 78)
print("2. EL DIPOLO SEGUN LA MASCARA")
n_cmb = unit(CMB_L, CMB_B)
resultados = {}
for nombre, sel, um, pesos_por_pixel in [("A (continua)", selA, uA, False),
                                          ("B (escalera)", selB, uB, True)]:
    if pesos_por_pixel:
        c = np.bincount(pix_a[sel], minlength=NP_A).astype(float)
        D = estimador(unit(l_a[maskB], b_a[maskB]), c[maskB], um)
    else:
        D = estimador(unit(gl[sel], gb[sel]), None, um)
    a = np.linalg.norm(D); l, b = vec2lb(D)
    resultados[nombre] = (D, a)
    print(f"   {nombre:<14} |D| = {a:.5f}   (l,b) = ({l:6.1f}, {b:+5.1f})   "
          f"sep = {sep_deg(l,b,CMB_L,CMB_B):5.1f}   par = {D @ n_cmb:.5f}")
aA, aB = resultados["A (continua)"][1], resultados["B (escalera)"][1]
print(f"\n   diferencia B - A = {100*(aB-aA)/aA:+.1f}%")

# =============================== 3. CUAL ESTA SESGADA ===============================
print("\n" + "=" * 78)
print("3. TEST DE SESGO — se inyectan dipolos CONOCIDOS y se mira cual los devuelve")
print(f"   {N_MOCKS_BIAS} simulaciones por amplitud, generadas a NSIDE={NSIDE_MOCK}\n")

NP_M = hp.nside2npix(NSIDE_MOCK)
l_m, b_m = centros(NSIDE_MOCK)
u_m = unit(l_m, b_m)
pix_m_a = hp.ang2pix(NSIDE_ANA, np.radians(90.0 - b_m), np.radians(l_m))
inA_m = geom_de(l_m, b_m) & ~malas_a[pix_m_a]        # region continua, sobre la rejilla mock
inB_m = maskB[pix_m_a]                                # region escalera
n_bar_m = selA.sum()/inA_m.sum()

print(f"   {'inyectado':>10} {'recuperado A':>24} {'sesgo A':>9} "
      f"{'recuperado B':>24} {'sesgo B':>9}")
bias = {}
for amp in (0.007, 0.014):
    rng = np.random.default_rng(int(amp*1e6))
    recA, recB = [], []
    for _ in range(N_MOCKS_BIAS):
        lam = np.clip(n_bar_m*(1.0 + u_m @ (amp*n_cmb)), 0, None)
        c_m = rng.poisson(lam).astype(float)
        # A: fuentes sobre la region continua, estimador sobre la rejilla fina
        DA = estimador(u_m[inA_m], c_m[inA_m], u_m[inA_m])
        # B: mismas fuentes agrupadas en pixeles gruesos, region escalera
        c_gr = np.bincount(pix_m_a[inB_m], weights=c_m[inB_m], minlength=NP_A)
        DB = estimador(unit(l_a[maskB], b_a[maskB]), c_gr[maskB], uB)
        recA.append(DA @ n_cmb); recB.append(DB @ n_cmb)
    recA, recB = np.array(recA), np.array(recB)
    bA, bB = 100*(recA.mean()-amp)/amp, 100*(recB.mean()-amp)/amp
    bias[amp] = (bA, bB)
    print(f"   {amp:10.5f} {recA.mean():12.5f} +/-{recA.std():8.5f} {bA:+8.1f}% "
          f"{recB.mean():12.5f} +/-{recB.std():8.5f} {bB:+8.1f}%")

mb_A = np.mean([v[0] for v in bias.values()])
mb_B = np.mean([v[1] for v in bias.values()])
print(f"\n   sesgo medio:  A = {mb_A:+.1f}%   B = {mb_B:+.1f}%")
fiducial = "A (continua)" if abs(mb_A) <= abs(mb_B) else "B (escalera)"
print(f"   -> FIDUCIAL: mascara {fiducial}")
print("   (si ambas salen insesgadas, la diferencia del 8% en los datos reales NO es")
print("    sesgo del estimador sino acoplamiento de multipolos altos reales a traves")
print("    del borde; entonces es sistematico legitimo y va al presupuesto de error)")

# =============================== 4. ESPECTRO Y SIGNIFICANCIA ===============================
print("\n" + "=" * 78)
print("4. SIGNIFICANCIA CON LA MASCARA FIDUCIAL")
D_fid, a_fid = resultados[fiducial]
par_fid = D_fid @ n_cmb

# C_l medido sobre la rejilla de analisis (mascara B por comodidad; el C_l es robusto)
cnt_cien = np.bincount(pix_a[cien & maskB[pix_a]], minlength=NP_A).astype(float)
nb = cnt_cien[maskB].sum()/maskB.sum()
delta = np.zeros(NP_A); delta[maskB] = cnt_cien[maskB]/nb - 1.0
LMAX = 3*NSIDE_ANA - 1
cl = np.maximum(hp.anafast(delta*maskB, lmax=LMAX)/maskB.mean() - (4*np.pi/NP_A)/nb, 0.0)
cl[0:2] = 0.0
print(f"   C_2 = {cl[2]:.3e}")

usar_A = fiducial.startswith("A")
rng = np.random.default_rng(2026)
a_kin, par_kin = [], []
for _ in range(N_MOCKS_SIG):
    try:
        d = hp.synfast(cl, NSIDE_MOCK, lmax=min(LMAX, 3*NSIDE_MOCK-1), verbose=False)
    except TypeError:
        d = hp.synfast(cl, NSIDE_MOCK, lmax=min(LMAX, 3*NSIDE_MOCK-1))
    d = d + u_m @ (D_ESPERADO*n_cmb)
    c_m = rng.poisson(np.clip(n_bar_m*(1.0+d), 0, None)).astype(float)
    if usar_A:
        D = estimador(u_m[inA_m], c_m[inA_m], u_m[inA_m])
    else:
        c_gr = np.bincount(pix_m_a[inB_m], weights=c_m[inB_m], minlength=NP_A)
        D = estimador(unit(l_a[maskB], b_a[maskB]), c_gr[maskB], uB)
    a_kin.append(np.linalg.norm(D)); par_kin.append(D @ n_cmb)
a_kin, par_kin = np.array(a_kin), np.array(par_kin)

s_a = (a_fid - a_kin.mean())/a_kin.std()
s_p = (par_fid - par_kin.mean())/par_kin.std()
print(f"\n   |D|      : {a_fid:.5f} contra {a_kin.mean():.5f} +/- {a_kin.std():.5f}"
      f"   {(a_kin >= a_fid).sum()}/{N_MOCKS_SIG}  ->  {s_a:.2f} sigma (solo estadistico)")
print(f"   paralela : {par_fid:.5f} contra {par_kin.mean():.5f} +/- {par_kin.std():.5f}"
      f"   {(par_kin >= par_fid).sum()}/{N_MOCKS_SIG}  ->  {s_p:.2f} sigma (solo estadistico)")

# =============================== 5. PRESUPUESTO DE ERROR ===============================
print("\n" + "=" * 78)
print("5. PRESUPUESTO DE ERROR Y RESULTADO FINAL")
sig_stat = a_kin.std()
sig_mask = abs(aB - aA)/2.0
print(f"\n   AMPLITUD  |D| = {a_fid:.5f}")
print(f"      estadistico (clustering + Poisson) : +/- {sig_stat:.5f}  ({100*sig_stat/a_fid:.1f}%)")
print(f"      sistematico de mascara             : +/- {sig_mask:.5f}  ({100*sig_mask/a_fid:.1f}%)")
tot = np.hypot(sig_stat, sig_mask)
print(f"      total en cuadratura                : +/- {tot:.5f}  ({100*tot/a_fid:.1f}%)")

coc = a_fid/D_ESPERADO
e_stat = sig_stat/D_ESPERADO
e_mask = sig_mask/D_ESPERADO
e_esp = coc*D_ESPERADO_SYS/D_ESPERADO
print(f"\n   COCIENTE sobre la expectativa cinematica = {coc:.2f}")
print(f"      +/- {e_stat:.2f} (estadistico)")
print(f"      +/- {e_mask:.2f} (mascara)")
print(f"      +/- {e_esp:.2f} (expectativa: x sin meseta, alpha incierto)")
e_tot = np.sqrt(e_stat**2 + e_mask**2 + e_esp**2)
print(f"      = {coc:.2f} +/- {e_tot:.2f}  (todo en cuadratura)")

s_final = (a_fid - a_kin.mean())/np.hypot(a_kin.std(), sig_mask)
sp_final = (par_fid - par_kin.mean())/np.hypot(par_kin.std(), sig_mask)
print(f"\n   SIGNIFICANCIA INCLUYENDO EL SISTEMATICO DE MASCARA")
print(f"      sobre |D|      : {s_final:.2f} sigma   (era {s_a:.2f} solo con estadistico)")
print(f"      sobre paralela : {sp_final:.2f} sigma   (era {s_p:.2f})")
print(f"\n   [Secrest et al. 2021: 4.9 sigma ; reevaluacion 2025: 3.27-3.63 sigma]")

# =============================== 6. FIGURA ===============================
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5), dpi=145)
x = np.arange(2)
amps = sorted(bias.keys())
axs[0].bar(x-0.18, [bias[a][0] for a in amps], 0.36, color='#1b6b50', label='máscara A')
axs[0].bar(x+0.18, [bias[a][1] for a in amps], 0.36, color='#b5423a', label='máscara B')
axs[0].axhline(0, color='k', lw=1.2)
axs[0].set_xticks(x); axs[0].set_xticklabels([f"{a:.3f}" for a in amps])
axs[0].set_xlabel("dipolo inyectado"); axs[0].set_ylabel("sesgo (%)")
axs[0].set_title("¿Cuál máscara está sesgada?"); axs[0].legend(fontsize=9); axs[0].grid(alpha=.3)

axs[1].hist(par_kin*100, bins=50, color='#1b6b50', alpha=.7)
axs[1].axvline(par_fid*100, color='#b5423a', lw=2.5, label=f'observado {par_fid*100:.2f}%')
axs[1].axvline(D_ESPERADO*100, color='k', ls='--', lw=1.5, label='esperado')
axs[1].set_xlabel("componente ∥ al CMB (%)"); axs[1].set_ylabel("simulaciones")
axs[1].set_title(f"Nula cinemática  ({sp_final:.1f}σ con sistemático)")
axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)

et = [100*e_stat/coc, 100*e_mask/coc, 100*e_esp/coc]
axs[2].barh(["estadístico", "máscara", "expectativa"], et,
            color=['#2e6fa8', '#c77a30', '#7a4fa3'])
for i, v in enumerate(et):
    axs[2].text(v+0.3, i, f"{v:.1f}%", va='center', fontsize=9)
axs[2].set_xlabel("contribución al error del cociente (%)")
axs[2].set_title(f"Presupuesto de error   ({coc:.2f} ± {e_tot:.2f})")
axs[2].grid(alpha=.3, axis='x')

plt.tight_layout(); plt.savefig("mascara_final.png", dpi=145)
print("\nFigura guardada: mascara_final.png")
plt.show()

print("""
COMO LEER EL TEST DE SESGO

 * Si una de las dos mascaras devuelve sistematicamente menos (o mas) de lo inyectado
   y la otra no, esa esta sesgada y la otra es la buena. El 8% de los datos reales era
   un error, y se corrige eligiendo la fiducial.
 * Si LAS DOS salen insesgadas con dipolos puros, entonces el 8% no es sesgo del
   estimador: viene de que el cielo real tiene multipolos altos que se acoplan de
   forma distinta a traves de un borde liso y de un borde dentado. En ese caso es un
   sistematico legitimo y su sitio es el presupuesto de error, que es lo que hace el
   apartado 5.

 En ninguno de los dos casos se puede reportar el numero sin ese termino. La barra
 que sale del apartado 5 es la honesta.
""")
