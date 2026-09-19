"""
quaia_dipolo_z.py
-----------------
Dipolo de cuasares en QUAIA, troceado por corrimiento al rojo.

POR QUE ESTO ES DISTINTO A CATWISE
  * Quaia tiene z. CatWISE no. Eso permite preguntar si el dipolo EVOLUCIA con el
    tiempo cosmico, que es una pregunta abierta y no una reproduccion.
  * Seleccion completamente distinta (Gaia optico + unWISE infrarrojo), asi que las
    sistematicas no son las mismas. Si la anomalia aparece en ambos catalogos con el
    mismo pipeline independiente, eso pesa mucho mas que cualquiera de los dos solo.

LA DIFERENCIA CRITICA DE METODO
  Con CatWISE construimos la mascara a mano (|b|>30 + Nubes + artefactos) y el
  catalogo aleatorio era uniforme sobre esa region. Con Quaia eso seria un ERROR:
  la ley de escaneo de Gaia imprime un patron angular fuerte. Aqui se usa el MAPA DE
  FUNCION DE SELECCION publicado por el equipo de Quaia, que da la probabilidad de
  deteccion pixel a pixel. Entra en el estimador como PESO de los randoms.

  Implementacion: en vez de generar randoms por Monte Carlo, se usan directamente los
  centros de pixel HEALPix con peso = valor de la funcion de seleccion. Es exacto y
  no introduce ruido de muestreo.

ARCHIVOS QUE NECESITAS (todos del mismo Zenodo, 10.5281/zenodo.10403370)
  quaia_G20.0.fits                        <- ya lo tienes
  selection_function_NSIDE64_G20.0.fits   <- IMPRESCINDIBLE, bajalo
  quaia_G20.5.fits                        <- opcional pero recomendado, ver abajo

  El G20.5 sirve para medir x en G=20.0 sin que el propio corte del catalogo trunque
  los conteos. Es exactamente la leccion del parche profundo de CatWISE. Sin el, el
  script hace un ajuste de un solo lado y te avisa de que el x resultante esta sesgado.

Requiere: numpy, matplotlib, astropy, healpy.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.table import Table

# =============================== CONFIGURACION ===============================
F_CAT   = "quaia_G20.0.fits"
F_SEL   = "selection_function_NSIDE64_G20.0.fits"
F_CAT_DEEP = "quaia_G20.5.fits"          # opcional, para medir x correctamente

G_CORTE   = 20.0          # limite de magnitud del catalogo
SEL_MIN   = 0.50          # umbral de completitud (la literatura usa ~0.5)
B_CUT     = 15.0          # mascara galactica extra: el equipo avisa que la funcion
                          # de seleccion es menos precisa cerca del plano
MASK_MC   = True          # enmascarar Nubes de Magallanes (tambien recomendado)
RAD_MC    = 8.0

N_BINS_Z  = 4             # numero de capas de z (por cuantiles: N similar en cada una)
ALPHA_OPT = 0.61          # indice espectral optico (S_nu ~ nu^-alpha). Ver nota al final.
N_BOOT    = 150

V_SOBRE_C = 369.82 / 299792.458
CMB_L, CMB_B = 264.021, 48.253

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

def vec2lb(v):
    n = v/np.linalg.norm(v)
    return np.degrees(np.arctan2(n[1], n[0])) % 360, np.degrees(np.arcsin(np.clip(n[2], -1, 1)))

def sep_deg(l1, b1, l2, b2):
    return np.degrees(np.arccos(np.clip(np.sum(unit(l1, b1)*unit(l2, b2), axis=-1), -1, 1)))

# =============================== ESTIMADOR CON PESOS ===============================
def fit_dipolo(u_d, u_r, w_r, pesos_d=None):
    """dN/dOmega ~ w(n) * (1 + D.n).  La funcion de seleccion entra como peso de los
    'randoms', que aqui son los centros de pixel. Devuelve D (vector) y su covarianza."""
    if pesos_d is None:
        mu_d = u_d.mean(0); n_ef = len(u_d)
        cov_mu = ((u_d.T @ u_d)/len(u_d) - np.outer(mu_d, mu_d))/len(u_d)
    else:
        W = pesos_d.sum()
        mu_d = (pesos_d[:, None]*u_d).sum(0)/W
        n_ef = W**2/(pesos_d**2).sum()
        cov_mu = ((pesos_d[:, None]*u_d).T @ u_d/W - np.outer(mu_d, mu_d))/n_ef
    Wr = w_r.sum()
    mu_r = (w_r[:, None]*u_r).sum(0)/Wr
    M = (w_r[:, None]*u_r).T @ u_r/Wr - np.outer(mu_r, mu_r)
    Minv = np.linalg.inv(M)
    D = Minv @ (mu_d - mu_r)
    return D, Minv @ cov_mu @ Minv.T, n_ef

def amp_err(D, covD):
    a = np.linalg.norm(D); n = D/a
    return a, np.sqrt(max(n @ covD @ n, 0.0))

# =============================== PENDIENTE LOCAL ===============================
def pendiente_local(mags, m0, ventana=0.35, un_lado=False):
    mags = np.sort(mags)
    lo, hi = (m0 - ventana, m0) if un_lado else (m0 - ventana, m0 + ventana)
    mm = np.linspace(lo, hi, 20)
    NN = np.searchsorted(mags, mm); g = NN > 30
    if g.sum() < 5:
        return np.nan
    return np.polyfit(mm[g], np.log10(NN[g]), 1)[0]/0.4

# =============================== CARGA ===============================
print("1. Catalogo")
if not os.path.exists(F_CAT):
    raise SystemExit(f"Falta {F_CAT}")
t = Table.read(F_CAT)
cols = {c.lower(): c for c in t.colnames}
print(f"   {len(t)} fuentes ; columnas: {', '.join(t.colnames[:12])}{'...' if len(t.colnames)>12 else ''}")

def buscar(*cands):
    for c in cands:
        if c in cols:
            return cols[c]
    return None

c_ra = buscar('ra'); c_dec = buscar('dec')
c_z  = buscar('redshift_quaia', 'z', 'redshift')
c_g  = buscar('phot_g_mean_mag', 'mag_g_gaia', 'g', 'gmag', 'phot_g_mean_mag_gaia')
if None in (c_ra, c_dec, c_z):
    raise SystemExit(f"No encuentro ra/dec/redshift. Columnas: {t.colnames}")
print(f"   usando: ra='{c_ra}' dec='{c_dec}' z='{c_z}' G='{c_g}'")

ra  = np.asarray(t[c_ra], float)
dec = np.asarray(t[c_dec], float)
zq  = np.asarray(t[c_z], float)
gm  = np.asarray(t[c_g], float) if c_g else np.full(len(ra), np.nan)
gl, gb = eq2gal(ra, dec)

# =============================== FUNCION DE SELECCION ===============================
print("\n2. Funcion de seleccion")
if not os.path.exists(F_SEL):
    print("   *** NO ENCONTRADA ***")
    print("   Bajala del mismo Zenodo (10.5281/zenodo.10403370).")
    print("   Sin ella, la ley de escaneo de Gaia domina el dipolo y el resultado no vale.")
    print("   Archivos .fits en esta carpeta:", glob.glob("*.fits"))
    raise SystemExit()
sel = hp.read_map(F_SEL, verbose=False) if 'verbose' in hp.read_map.__code__.co_varnames \
      else hp.read_map(F_SEL)
nside = hp.npix2nside(len(sel))
sel = np.nan_to_num(sel, nan=0.0)
print(f"   mapa NSIDE={nside}, {len(sel)} pixeles, rango [{sel.min():.3f}, {sel.max():.3f}]")

# ¿el mapa esta en coordenadas galacticas o ecuatoriales? La seleccion debe CAER
# cerca del plano galactico (polvo + estrellas). Probamos las dos hipotesis.
th, ph = hp.pix2ang(nside, np.arange(len(sel)))
lat_map = 90.0 - np.degrees(th); lon_map = np.degrees(ph)
def contraste(lat_gal):
    a = sel[np.abs(lat_gal) < 10].mean(); b = sel[np.abs(lat_gal) > 50].mean()
    return b/max(a, 1e-6)
c_gal = contraste(lat_map)                                   # hipotesis: mapa en galacticas
_, lat_si_eq = eq2gal(lon_map, lat_map)                      # hipotesis: mapa en ecuatoriales
c_eq = contraste(lat_si_eq)
MAPA_GALACTICO = c_gal >= c_eq
print(f"   contraste polo/plano: si galactico={c_gal:.2f}, si ecuatorial={c_eq:.2f}"
      f"  ->  mapa interpretado como {'GALACTICO' if MAPA_GALACTICO else 'ECUATORIAL'}")

if MAPA_GALACTICO:
    l_pix, b_pix = lon_map, lat_map
    pix_de_fuente = hp.ang2pix(nside, np.radians(90.0 - gb), np.radians(gl))
else:
    l_pix, b_pix = eq2gal(lon_map, lat_map)
    pix_de_fuente = hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra))

# =============================== MASCARA ===============================
print("\n3. Mascara")
ok_pix = (sel > SEL_MIN) & (np.abs(b_pix) > B_CUT)
LMC, SMC = (80.894, -69.756), (13.187, -72.829)
if MASK_MC:
    lmc_l, lmc_b = eq2gal(*LMC); smc_l, smc_b = eq2gal(*SMC)
    ok_pix &= (sep_deg(l_pix, b_pix, lmc_l, lmc_b) > RAD_MC)
    ok_pix &= (sep_deg(l_pix, b_pix, smc_l, smc_b) > RAD_MC)
f_sky = ok_pix.sum()/len(sel)
ok_src = ok_pix[pix_de_fuente] & np.isfinite(zq)
print(f"   sel>{SEL_MIN}, |b|>{B_CUT}, Nubes fuera -> f_sky={f_sky:.3f}, "
      f"{ok_src.sum()} de {len(ra)} fuentes ({100*ok_src.mean():.1f}%)")

u_r = unit(l_pix[ok_pix], b_pix[ok_pix])
w_r = sel[ok_pix]

# =============================== x(G) ===============================
print("\n4. Pendiente de conteos x")
if os.path.exists(F_CAT_DEEP):
    td = Table.read(F_CAT_DEEP)
    cd = {c.lower(): c for c in td.colnames}
    gcol = cd.get('phot_g_mean_mag') or cd.get('mag_g_gaia') or cd.get('g')
    zcol = cd.get('redshift_quaia') or cd.get('z')
    g_deep = np.asarray(td[gcol], float); z_deep = np.asarray(td[zcol], float)
    print(f"   usando {F_CAT_DEEP} ({len(td)} fuentes) -> x sin truncamiento")
    UN_LADO = False
else:
    g_deep, z_deep = gm, zq
    print(f"   {F_CAT_DEEP} no encontrado: ajuste de UN SOLO LADO sobre G<{G_CORTE}.")
    print("   *** El x resultante esta sesgado hacia abajo. Baja el G20.5 si puedes. ***")
    UN_LADO = True

# =============================== CAPAS DE Z ===============================
print(f"\n5. DIPOLO POR CAPA DE REDSHIFT ({N_BINS_Z} capas por cuantiles)")
zz = zq[ok_src]
bordes = np.quantile(zz, np.linspace(0, 1, N_BINS_Z + 1))
bordes[0] -= 1e-6; bordes[-1] += 1e-6
rng = np.random.default_rng(0)

u_all = unit(gl[ok_src], gb[ok_src])
g_all = gm[ok_src]
n_cmb = unit(CMB_L, CMB_B)

print(f"   {'capa z':>14} {'N':>8} {'x':>6} {'D_esp':>8} {'|D|':>17} "
      f"{'l':>7} {'b':>7} {'sep':>6} {'coc':>12}")
res = []
for k in range(N_BINS_Z):
    m = (zz > bordes[k]) & (zz <= bordes[k+1])
    u_k = u_all[m]
    D, covD, nef = fit_dipolo(u_k, u_r, w_r)
    amp, aerr = amp_err(D, covD)
    l, b = vec2lb(D); s = sep_deg(l, b, CMB_L, CMB_B)

    md = (z_deep > bordes[k]) & (z_deep <= bordes[k+1]) & np.isfinite(g_deep)
    x_k = pendiente_local(g_deep[md], G_CORTE, un_lado=UN_LADO)
    D_esp = (2 + x_k*(1 + ALPHA_OPT))*V_SOBRE_C
    coc, cerr = amp/D_esp, aerr/D_esp
    print(f"   {bordes[k]:5.2f}-{bordes[k+1]:5.2f} {m.sum():8d} {x_k:6.3f} {D_esp:8.5f} "
          f"{amp:.5f}+/-{aerr:.5f} {l:7.1f} {b:+7.1f} {s:6.1f} {coc:5.2f}+/-{cerr:4.2f}")
    res.append((0.5*(bordes[k]+bordes[k+1]), m.sum(), x_k, D_esp, amp, aerr, l, b, s, coc, cerr))

# muestra completa, como control contra la literatura
D, covD, _ = fit_dipolo(u_all, u_r, w_r)
amp, aerr = amp_err(D, covD); l, b = vec2lb(D)
x_tot = pendiente_local(g_deep[np.isfinite(g_deep)], G_CORTE, un_lado=UN_LADO)
D_esp_tot = (2 + x_tot*(1 + ALPHA_OPT))*V_SOBRE_C
print(f"\n   MUESTRA COMPLETA: N={ok_src.sum()}  x={x_tot:.3f}  D_esp={D_esp_tot:.5f}")
print(f"      |D|={amp:.5f}+/-{aerr:.5f}  (l,b)=({l:.1f},{b:+.1f})  "
      f"sep={sep_deg(l,b,CMB_L,CMB_B):.1f}  cociente={amp/D_esp_tot:.2f}+/-{aerr/D_esp_tot:.2f}")

# =============================== ¿HAY EVOLUCION? ===============================
res = np.array(res, dtype=float)
zc, Nk, xk, De, Dk, Dke, lk, bk, sk, ck, cke = res.T
w = 1/cke**2
cm = np.sum(ck*w)/np.sum(w); chi2 = np.sum((ck-cm)**2*w); dof = N_BINS_Z - 1
print(f"\n6. ¿EL COCIENTE EVOLUCIONA CON z?")
print(f"   cociente medio = {cm:.2f}")
print(f"   chi2 contra constante = {chi2:.2f} / {dof} g.l.  (chi2/dof = {chi2/dof:.2f})")
print("   Las capas de z son DISJUNTAS, asi que aqui el chi2 si es directamente valido")
print("   (a diferencia de los cortes de magnitud anidados de CatWISE).")
if chi2/dof > 2.5:
    print("   -> hay indicios de evolucion con z. Verifica antes de creertelo:")
    print("      los errores de z de Quaia mezclan fuentes entre capas.")
else:
    print("   -> compatible con un cociente independiente de z.")

# =============================== FIGURA ===============================
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), dpi=145)
axs[0].errorbar(zc, Dk*100, yerr=Dke*100, fmt='o-', color='#b5423a', lw=2, ms=7, capsize=4,
                label=r'$D_{obs}$')
axs[0].plot(zc, De*100, 's--', color='0.35', lw=2, ms=5, label=r'$D_{esp}$')
axs[0].set_xlabel("z"); axs[0].set_ylabel("amplitud (%)")
axs[0].set_title("Dipolo por capa de z"); axs[0].legend(fontsize=9); axs[0].grid(alpha=.3)

axs[1].errorbar(zc, sk, yerr=np.degrees(Dke/Dk), fmt='o-', color='#c77a30', lw=2, ms=7, capsize=4)
axs[1].axhline(0, color='k', ls=':', lw=1.5, label='dirección del CMB')
axs[1].set_xlabel("z"); axs[1].set_ylabel("separación del CMB (grados)")
axs[1].set_title("¿Apunta siempre al mismo sitio?"); axs[1].legend(fontsize=9); axs[1].grid(alpha=.3)

axs[2].errorbar(zc, ck, yerr=cke, fmt='o-', color='#1b6b50', lw=2.2, ms=8, capsize=4)
axs[2].axhline(cm, color='#1b6b50', ls='--', lw=1.2, alpha=.7, label=f'media {cm:.2f}')
axs[2].axhline(1.0, color='k', ls=':', lw=1.5, label='sin anomalía')
axs[2].set_xlabel("z"); axs[2].set_ylabel(r"$D_{obs}/D_{esp}$")
axs[2].set_title(f"LA PREGUNTA NUEVA  ($\\chi^2$/gl = {chi2/dof:.2f})")
axs[2].legend(fontsize=8.5); axs[2].grid(alpha=.3); axs[2].set_ylim(0, max(4, ck.max()*1.3))
plt.tight_layout(); plt.savefig("quaia_dipolo_z.png", dpi=145)
print("\nFigura guardada: quaia_dipolo_z.png")
plt.show()

# =============================== NOTAS ===============================
print("""
LIMITACIONES QUE DEBES TENER PRESENTES AL LEER ESTO

 1. alpha esta FIJO en %.2f. En la banda G el indice espectral efectivo cambia con z,
    porque al desplazarse el espectro entran distintas partes del continuo y lineas
    de emision en el filtro. Si el cociente sale dependiente de z, descarta primero
    que sea esto: repite con alpha variando entre 0.4 y 0.9 y mira si el efecto
    sobrevive.

 2. Los redshifts de Quaia no son espectroscopicos. En G<20.0 hay del orden de 10%%
    de errores catastroficos, lo que MEZCLA fuentes entre capas y DILUYE cualquier
    evolucion real. O sea: el test es conservador. Si ves evolucion pese a esto,
    es mas interesante, no menos.

 3. Repite con bordes de capa distintos (cambia N_BINS_Z, o usa bordes fijos en vez
    de cuantiles). Una señal que depende de donde pongas los bordes no es una señal.

 4. Las barras son de muestreo. El clustering de cuasares añade varianza y aqui pesa
    MAS que en CatWISE, porque hay menos fuentes por capa.
""" % ALPHA_OPT)
