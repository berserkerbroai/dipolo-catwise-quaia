"""
reconciliar.py
--------------
El estimador de PUNTOS da |D| = 0.0139 y el de PIXELES 0.01843 sobre la misma muestra.
33% de discrepancia. El test de recuperacion demostro que el estimador de pixeles NO
esta sesgado, asi que el problema esta en la muestra o en la mascara, no en la
matematica. Este script lo aisla variando una cosa a la vez.

CANDIDATOS, del mas probable al menos:

  A. RECHAZO DE ARTEFACTOS. En test_final.py se calculaba sobre la muestra PROFUNDA
     (W1<16.75, 2.58M fuentes, celdas de 0.32 deg^2) y se aplicaba a todo: 50 celdas,
     14331 fuentes. En clustering.py se recalcula sobre la muestra final ya recortada,
     en pixeles HEALPix de 0.84 deg^2: solo 14 pixeles. Pixeles mas grandes diluyen
     cada artefacto, y una muestra menor los hace menos visibles sobre Poisson.

  B. DEFINICION DE LA MASCARA EN EL BORDE. Con puntos, |b|>30 se evalua fuente por
     fuente. Con pixeles, se evalua en el CENTRO del pixel y las fuentes se asignan a
     el. Un pixel cuyo centro cae en b=29.8 se descarta entero aunque parte de su area
     este por encima de 30.

  C. PIXELIZACION DE LAS POSICIONES. Sustituir cada fuente por el centro de su pixel
     introduce un desplazamiento de ~0.5 grados. Deberia promediarse a cero sobre
     748 mil fuentes, pero conviene medirlo en vez de suponerlo.

Requiere: numpy, healpy.
"""
import numpy as np
import healpy as hp

CACHE = "cielo_cache.npz"
NSIDE = 64
B_CUT, RAD_MC, K_ART = 30.0, 8.0, 12.0
COLOR_MIN, COLOR_MAX, W1_MAX = 0.80, 1.60, 16.00
W1_PROFUNDO = 16.75          # el corte sobre el que se calculaba el rechazo antes
NB_FINA = 360                # la rejilla equiareal original
N_RANDOM = 6_000_000
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

def dipolo(u_d, u_r, pesos_d=None):
    if pesos_d is None:
        mu_d = u_d.mean(0)
    else:
        mu_d = (pesos_d[:, None]*u_d).sum(0)/pesos_d.sum()
    mu_r = u_r.mean(0)
    M = (u_r.T @ u_r)/len(u_r) - np.outer(mu_r, mu_r)
    return np.linalg.solve(M, mu_d - mu_r)

def informe(nombre, D, n):
    a = np.linalg.norm(D); l, b = vec2lb(D)
    print(f"   {nombre:<48} N={n:8d}  |D|={a:.5f}  (l,b)=({l:6.1f},{b:+5.1f})  "
          f"sep={sep_deg(l,b,CMB_L,CMB_B):5.1f}")
    return a

# =============================== CARGA ===============================
print("=" * 94)
z = np.load(CACHE)
ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
gl, gb = eq2gal(ra, dec); color = w1 - w2
print(f"Catalogo: {len(ra)} fuentes")

geom = ((np.abs(gb) > B_CUT)
        & (sep_deg(ra, dec, *LMC) > RAD_MC) & (sep_deg(ra, dec, *SMC) > RAD_MC))
ciencia = geom & (color >= COLOR_MIN) & (color < COLOR_MAX) & (w1 < W1_MAX)
print(f"Muestra de ciencia antes de rechazar artefactos: {ciencia.sum()}")

# randoms con la mascara geometrica (sin rechazo todavia)
rng = np.random.default_rng(1)
dr = np.degrees(np.arcsin(rng.uniform(-1, 1, N_RANDOM))); rr = rng.uniform(0, 360, N_RANDOM)
glr, gbr = eq2gal(rr, dr)
geom_r = ((np.abs(gbr) > B_CUT)
          & (sep_deg(rr, dr, *LMC) > RAD_MC) & (sep_deg(rr, dr, *SMC) > RAD_MC))

# =============================== LAS DOS MASCARAS DE ARTEFACTOS ===============================
def celda_fina(r, d):
    i = np.clip(((np.sin(np.radians(d))+1)/2*NB_FINA).astype(int), 0, NB_FINA-1)
    j = np.clip((r/360*NB_FINA).astype(int), 0, NB_FINA-1)
    return i*NB_FINA + j

# (A1) la original: rejilla fina, calculada sobre la muestra PROFUNDA
cid_d, cid_r = celda_fina(ra, dec), celda_fina(rr, dr)
prof = geom & (w1 < W1_PROFUNDO)
hb = np.bincount(cid_d[prof], minlength=NB_FINA**2).astype(float)
mu_f = hb[hb > 0].mean()
malas_f = np.where(hb > mu_f + K_ART*np.sqrt(mu_f))[0]
art_fina_d = np.isin(cid_d, malas_f); art_fina_r = np.isin(cid_r, malas_f)
print(f"\n(A1) rejilla fina sobre muestra profunda: {len(malas_f)} celdas, "
      f"{art_fina_d.sum()} fuentes del catalogo, {(art_fina_d & ciencia).sum()} de la de ciencia")

# (A2) la de clustering.py: HEALPix, calculada sobre la muestra FINAL
NPIX = hp.nside2npix(NSIDE)
pix_d = hp.ang2pix(NSIDE, np.radians(90.0 - gb), np.radians(gl))
pix_r = hp.ang2pix(NSIDE, np.radians(90.0 - gbr), np.radians(glr))
th, ph = hp.pix2ang(NSIDE, np.arange(NPIX))
l_pix, b_pix = np.degrees(ph), 90.0 - np.degrees(th)
lmc_l, lmc_b = eq2gal(*LMC); smc_l, smc_b = eq2gal(*SMC)
mask_pix = ((np.abs(b_pix) > B_CUT)
            & (sep_deg(l_pix, b_pix, lmc_l, lmc_b) > RAD_MC)
            & (sep_deg(l_pix, b_pix, smc_l, smc_b) > RAD_MC))
cnt_fin = np.bincount(pix_d[ciencia], minlength=NPIX).astype(float)
ocup = cnt_fin[mask_pix & (cnt_fin > 0)]
mu_h = ocup.mean()
malas_h = cnt_fin > mu_h + K_ART*np.sqrt(mu_h)
print(f"(A2) HEALPix sobre muestra final:          {malas_h.sum()} pixeles, "
      f"{np.isin(pix_d, np.where(malas_h)[0]).sum()} fuentes del catalogo, "
      f"{(np.isin(pix_d, np.where(malas_h)[0]) & ciencia).sum()} de la de ciencia")

# (A3) HEALPix pero calculada sobre la muestra PROFUNDA, que es lo correcto
cnt_prof = np.bincount(pix_d[prof], minlength=NPIX).astype(float)
ocup2 = cnt_prof[mask_pix & (cnt_prof > 0)]
mu_h2 = ocup2.mean()
malas_h2 = cnt_prof > mu_h2 + K_ART*np.sqrt(mu_h2)
print(f"(A3) HEALPix sobre muestra profunda:       {malas_h2.sum()} pixeles, "
      f"{np.isin(pix_d, np.where(malas_h2)[0]).sum()} fuentes del catalogo, "
      f"{(np.isin(pix_d, np.where(malas_h2)[0]) & ciencia).sum()} de la de ciencia")

# =============================== EXPERIMENTOS ===============================
print("\n" + "=" * 94)
print("EXPERIMENTO 1 — ESTIMADOR DE PUNTOS, variando solo el rechazo de artefactos")
print("   (la referencia publicada en el documento es 0.01389)\n")
for nombre, bad_d, bad_r in [
    ("sin rechazo",                          np.zeros(len(ra), bool), np.zeros(N_RANDOM, bool)),
    ("A1: rejilla fina / muestra profunda",  art_fina_d,              art_fina_r),
    ("A2: HEALPix / muestra final",          np.isin(pix_d, np.where(malas_h)[0]),
                                             np.isin(pix_r, np.where(malas_h)[0])),
    ("A3: HEALPix / muestra profunda",       np.isin(pix_d, np.where(malas_h2)[0]),
                                             np.isin(pix_r, np.where(malas_h2)[0])),
]:
    kd = ciencia & ~bad_d; kr = geom_r & ~bad_r
    D = dipolo(unit(gl[kd], gb[kd]), unit(glr[kr], gbr[kr]))
    informe(nombre, D, kd.sum())

print("\nEXPERIMENTO 2 — MISMO rechazo (A1), comparando los tres estimadores")
print("   Si los tres coinciden, la pixelizacion no es el problema.\n")
kd = ciencia & ~art_fina_d
kr = geom_r & ~art_fina_r
D1 = dipolo(unit(gl[kd], gb[kd]), unit(glr[kr], gbr[kr]))
informe("puntos, mascara evaluada por fuente", D1, kd.sum())

# posiciones pixelizadas, pero mascara aun por fuente
D2 = dipolo(unit(l_pix[pix_d[kd]], b_pix[pix_d[kd]]), unit(glr[kr], gbr[kr]))
informe("posiciones pixelizadas, mascara por fuente", D2, kd.sum())

# todo por pixel: conteos y mascara en centros de pixel
mask_final = mask_pix & ~np.isin(np.arange(NPIX), np.unique(pix_d[art_fina_d]))
kd_pix = kd & mask_final[pix_d]
cnt = np.bincount(pix_d[kd_pix], minlength=NPIX).astype(float)
sel_pix = mask_final & (cnt >= 0)
D3 = dipolo(unit(l_pix[sel_pix], b_pix[sel_pix]), unit(l_pix[sel_pix], b_pix[sel_pix]),
            pesos_d=cnt[sel_pix])
informe("conteos por pixel, mascara en centros", D3, int(cnt.sum()))

print("\nEXPERIMENTO 3 — EFECTO DEL BORDE DE LA MASCARA")
print("   Se aleja el corte galactico del borde para ver si la discrepancia se cierra.\n")
for bc in (30.0, 32.0, 35.0):
    g2 = ((np.abs(gb) > bc) & (sep_deg(ra, dec, *LMC) > RAD_MC)
          & (sep_deg(ra, dec, *SMC) > RAD_MC))
    g2r = ((np.abs(gbr) > bc) & (sep_deg(rr, dr, *LMC) > RAD_MC)
           & (sep_deg(rr, dr, *SMC) > RAD_MC))
    c2 = g2 & (color >= COLOR_MIN) & (color < COLOR_MAX) & (w1 < W1_MAX) & ~art_fina_d
    kr2 = g2r & ~art_fina_r
    Dp = dipolo(unit(gl[c2], gb[c2]), unit(glr[kr2], gbr[kr2]))
    mp = ((np.abs(b_pix) > bc) & (sep_deg(l_pix, b_pix, lmc_l, lmc_b) > RAD_MC)
          & (sep_deg(l_pix, b_pix, smc_l, smc_b) > RAD_MC)
          & ~np.isin(np.arange(NPIX), np.unique(pix_d[art_fina_d])))
    cx = np.bincount(pix_d[c2 & mp[pix_d]], minlength=NPIX).astype(float)
    Dx = dipolo(unit(l_pix[mp], b_pix[mp]), unit(l_pix[mp], b_pix[mp]), pesos_d=cx[mp])
    ap = np.linalg.norm(Dp); ax = np.linalg.norm(Dx)
    print(f"   |b|>{bc:4.1f}:  puntos {ap:.5f}   pixeles {ax:.5f}   "
          f"diferencia {100*(ax-ap)/ap:+6.1f}%")

print("""
COMO LEERLO

 * Si en el EXPERIMENTO 1 la fila A1 reproduce 0.01389 y la A2 da 0.0184, el culpable
   es el rechazo de artefactos y la correccion es calcularlo siempre sobre la muestra
   profunda. En ese caso hay que rehacer clustering.py con la mascara A3 y volver a
   sacar la significancia.
 * Si en el EXPERIMENTO 2 los tres estimadores coinciden, la pixelizacion esta
   descartada y el problema es solo de mascara.
 * Si en el EXPERIMENTO 3 la diferencia se encoge al alejar el corte del borde, es
   efecto de borde: pixeles partidos por |b|=30. Se arregla enmascarando tambien los
   pixeles cuyo centro esta a menos de un radio de pixel del limite.
 * Si nada de esto la cierra, el siguiente sospechoso es la fuga del cuadrupolo: C_2
   salio 12 veces el ruido de disparo, y las dos definiciones de mascara acoplan ese
   cuadrupolo al dipolo de forma ligeramente distinta.
""")
