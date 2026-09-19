"""
barrido_artefactos.py
---------------------
El ultimo termino del presupuesto de error.

El rechazo de artefactos mueve la amplitud un 12.9% entre el criterio global (12
sigma sobre rejilla NB=360) y el local (8 sigma, haz de 5 grados, NSIDE=128). Ese
sistematico esta estimado con DOS puntos, que es lo mismo que no estimarlo.

Este script barre los dos parametros libres del criterio local:
    K_ART       umbral en sigmas locales
    FWHM_SUAVE  escala angular del entorno que define la referencia

y mide cuanto se mueve el dipolo. Incluye ademas el criterio global y el caso SIN
rechazo, como extremos del rango.

COMO INTERPRETARLO
  * Si dentro de un rango razonable de parametros la amplitud se queda dentro del 3%,
    el rechazo es robusto y el 12.9% era la diferencia entre dos metodos, uno mejor
    que el otro. El sistematico es pequeño.
  * Si la amplitud se mueve tanto como el parametro, el rechazo es un sistematico del
    mismo orden que la senal y hay que reportarlo asi, no esconderlo.

Requiere: numpy, matplotlib, healpy.
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

CACHE = "cielo_cache.npz"
NSIDE_ANA, NSIDE_FINA = 128, 512
B_CUT, RAD_MC = 30.0, 8.0
COLOR_MIN, COLOR_MAX, W1_MAX = 0.80, 1.60, 16.00
W1_PROFUNDO = 16.75
D_ESPERADO = 0.00670
SIG_STAT = 0.00247            # de mascara_final.py
KS    = [6.0, 8.0, 10.0, 12.0, 16.0, 25.0]
FWHMS = [3.0, 5.0, 8.0, 12.0]
NB_GRUESA = 360               # rejilla del criterio global antiguo
K_GLOBAL = 12.0

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

def geom_de(lon, lat):
    ll, lb = eq2gal(*LMC); sl, sb = eq2gal(*SMC)
    return ((np.abs(lat) > B_CUT) & (sep_deg(lon, lat, ll, lb) > RAD_MC)
            & (sep_deg(lon, lat, sl, sb) > RAD_MC))

def centros(ns):
    th, ph = hp.pix2ang(ns, np.arange(hp.nside2npix(ns)))
    return np.degrees(ph), 90.0 - np.degrees(th)

def suavizar(m, fwhm):
    try:
        return hp.smoothing(m, fwhm=np.radians(fwhm), verbose=False)
    except TypeError:
        return hp.smoothing(m, fwhm=np.radians(fwhm))

# =============================== DATOS ===============================
print("=" * 80)
z = np.load(CACHE)
ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
gl, gb = eq2gal(ra, dec); color = w1 - w2
cien = (color >= COLOR_MIN) & (color < COLOR_MAX) & (w1 < W1_MAX)

NP_A = hp.nside2npix(NSIDE_ANA)
pix_a = hp.ang2pix(NSIDE_ANA, np.radians(90.0 - gb), np.radians(gl))
l_a, b_a = centros(NSIDE_ANA); geom_a = geom_de(l_a, b_a)
cnt_prof = np.bincount(pix_a[w1 < W1_PROFUNDO], minlength=NP_A).astype(float)

l_f, b_f = centros(NSIDE_FINA)
pix_f_a = hp.ang2pix(NSIDE_ANA, np.radians(90.0 - b_f), np.radians(l_f))
geom_f = geom_de(l_f, b_f)
geom_src = geom_de(gl, gb)
n_cmb = unit(CMB_L, CMB_B)

def dipolo_con(malas):
    """Mascara A (continua): fuentes por su propia posicion, region en rejilla fina."""
    sel = cien & geom_src & ~malas[pix_a]
    um = unit(l_f[geom_f & ~malas[pix_f_a]], b_f[geom_f & ~malas[pix_f_a]])
    ud = unit(gl[sel], gb[sel])
    mu_d, mu_r = ud.mean(0), um.mean(0)
    M = (um.T @ um)/len(um) - np.outer(mu_r, mu_r)
    D = np.linalg.solve(M, mu_d - mu_r)
    return D, sel.sum()

# referencia local, cacheando el suavizado por FWHM
cache_ref = {}
def malas_local(K, fwhm):
    if fwhm not in cache_ref:
        mf = geom_a.astype(float)
        cs, ms = suavizar(cnt_prof*mf, fwhm), suavizar(mf, fwhm)
        cache_ref[fwhm] = np.where(ms > 0.2, cs/np.maximum(ms, 1e-6), np.nan)
    ref = cache_ref[fwhm]
    exc = np.where(np.isfinite(ref) & (ref > 0),
                   (cnt_prof - ref)/np.sqrt(np.maximum(ref, 1.0)), 0.0)
    return geom_a & (exc > K)

# =============================== BARRIDO ===============================
print("BARRIDO DEL RECHAZO DE ARTEFACTOS  (mascara A fiducial)\n")
print(f"   {'FWHM':>6} {'K':>6} {'pix':>6} {'N':>9} {'|D|':>9} {'l':>7} {'b':>7} "
      f"{'sep':>6} {'par':>9}")
res = []
for fw in FWHMS:
    for K in KS:
        m = malas_local(K, fw)
        D, n = dipolo_con(m)
        a = np.linalg.norm(D); l, b = vec2lb(D)
        print(f"   {fw:6.1f} {K:6.1f} {m.sum():6d} {n:9d} {a:9.5f} {l:7.1f} {b:+7.1f} "
              f"{sep_deg(l,b,CMB_L,CMB_B):6.1f} {D @ n_cmb:9.5f}")
        res.append((fw, K, m.sum(), n, a, sep_deg(l, b, CMB_L, CMB_B), D @ n_cmb))
    print()
res = np.array(res)

# extremos: sin rechazo, y criterio global antiguo
sin_r, n0 = dipolo_con(np.zeros(NP_A, bool))
a0 = np.linalg.norm(sin_r); l0, b0 = vec2lb(sin_r)
print(f"   {'SIN RECHAZO':>13} {0:6d} {n0:9d} {a0:9.5f} {l0:7.1f} {b0:+7.1f} "
      f"{sep_deg(l0,b0,CMB_L,CMB_B):6.1f} {sin_r @ n_cmb:9.5f}")

def celda_gruesa(r, d):
    i = np.clip(((np.sin(np.radians(d))+1)/2*NB_GRUESA).astype(int), 0, NB_GRUESA-1)
    j = np.clip((r/360*NB_GRUESA).astype(int), 0, NB_GRUESA-1)
    return i*NB_GRUESA + j
cid = celda_gruesa(ra, dec)
hb = np.bincount(cid[geom_src & (w1 < W1_PROFUNDO)], minlength=NB_GRUESA**2).astype(float)
mu_g = hb[hb > 0].mean()
cel_malas = np.where(hb > mu_g + K_GLOBAL*np.sqrt(mu_g))[0]
mal_src = np.isin(cid, cel_malas)
sel_g = cien & geom_src & ~mal_src
cid_f = celda_gruesa(*((lambda l, b: (l, b))(*[np.degrees(x) for x in (0, 0)])) ) if False else None
# region fina equivalente: se quitan las celdas malas tambien del area
l_fe, d_fe = None, None
ra_f, dec_f = None, None
# para la region, usamos la rejilla fina en galacticas -> ecuatoriales no hace falta:
# aproximamos quitando de la rejilla fina los pixeles cuyo centro cae en celda mala
th_f, ph_f = hp.pix2ang(NSIDE_FINA, np.arange(hp.nside2npix(NSIDE_FINA)))
# convertir centros finos a ecuatoriales para evaluar la celda gruesa
def gal2eq(l, b):
    l, b = np.radians(l), np.radians(b)
    sd = np.sin(b)*np.sin(DEC_NGP) + np.cos(b)*np.cos(DEC_NGP)*np.cos(L_NCP-l)
    de = np.arcsin(np.clip(sd, -1, 1))
    y = np.cos(b)*np.sin(L_NCP-l)
    x = np.sin(b)*np.cos(DEC_NGP) - np.cos(b)*np.sin(DEC_NGP)*np.cos(L_NCP-l)
    return np.degrees(RA_NGP + np.arctan2(y, x)) % 360, np.degrees(de)
ra_fi, dec_fi = gal2eq(l_f, b_f)
mal_fina = np.isin(celda_gruesa(ra_fi, dec_fi), cel_malas)
um_g = unit(l_f[geom_f & ~mal_fina], b_f[geom_f & ~mal_fina])
ud_g = unit(gl[sel_g], gb[sel_g])
mu_d, mu_r = ud_g.mean(0), um_g.mean(0)
Mg = (um_g.T @ um_g)/len(um_g) - np.outer(mu_r, mu_r)
Dg = np.linalg.solve(Mg, mu_d - mu_r)
ag = np.linalg.norm(Dg); lg, bg = vec2lb(Dg)
print(f"   {'GLOBAL 12sig':>13} {len(cel_malas):6d} {sel_g.sum():9d} {ag:9.5f} {lg:7.1f} "
      f"{bg:+7.1f} {sep_deg(lg,bg,CMB_L,CMB_B):6.1f} {Dg @ n_cmb:9.5f}")

# =============================== SISTEMATICO ===============================
print("\n" + "=" * 80)
print("SISTEMATICO DEL RECHAZO")
amps = res[:, 4]
razonable = res[(res[:, 1] >= 6) & (res[:, 1] <= 16)][:, 4]
print(f"   barrido completo    : {amps.min():.5f} a {amps.max():.5f}  "
      f"(dispersion {100*amps.std()/amps.mean():.1f}%)")
print(f"   K entre 6 y 16      : {razonable.min():.5f} a {razonable.max():.5f}  "
      f"(dispersion {100*razonable.std()/razonable.mean():.1f}%)")
print(f"   sin rechazo         : {a0:.5f}")
print(f"   criterio global 12s : {ag:.5f}")
todos = np.concatenate([razonable, [ag]])
s_art = todos.std()
print(f"\n   -> sistematico adoptado (desv. tipica incluyendo el criterio global): "
      f"{s_art:.5f}  ({100*s_art/razonable.mean():.1f}%)")

a_fid = res[(res[:, 0] == 5.0) & (res[:, 1] == 8.0)][0, 4]
par_fid = res[(res[:, 0] == 5.0) & (res[:, 1] == 8.0)][0, 6]
s_mask = 0.00032
tot = np.sqrt(SIG_STAT**2 + s_mask**2 + s_art**2)
print(f"\n   PRESUPUESTO FINAL sobre |D| = {a_fid:.5f}")
for nm, v in [("estadistico", SIG_STAT), ("artefactos", s_art), ("mascara", s_mask)]:
    print(f"      {nm:<13} +/- {v:.5f}  ({100*v/a_fid:4.1f}%)")
print(f"      {'total':<13} +/- {tot:.5f}  ({100*tot/a_fid:4.1f}%)")
par_mu, par_sd = 0.00663, 0.00262
print(f"\n   significancia (paralela), con todo: "
      f"{(par_fid-par_mu)/np.sqrt(par_sd**2+s_mask**2+s_art**2):.2f} sigma")
coc = a_fid/D_ESPERADO
e = np.sqrt((SIG_STAT/D_ESPERADO)**2 + (s_mask/D_ESPERADO)**2 +
            (s_art/D_ESPERADO)**2 + (coc*0.10)**2)
print(f"   cociente: {coc:.2f} +/- {e:.2f}")

# =============================== FIGURA ===============================
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5), dpi=145)
col = plt.cm.viridis(np.linspace(0, .85, len(FWHMS)))
for i, fw in enumerate(FWHMS):
    s = res[res[:, 0] == fw]
    axs[0].plot(s[:, 1], s[:, 4]*100, '-o', color=col[i], lw=2, ms=6, label=f"{fw:.0f}°")
    axs[1].plot(s[:, 1], s[:, 5], '-o', color=col[i], lw=2, ms=6)
axs[0].axhline(a0*100, color='k', ls=':', lw=1.5, label='sin rechazo')
axs[0].axhline(ag*100, color='#b5423a', ls='--', lw=1.5, label='global 12σ')
axs[0].set_xlabel("K (σ local)"); axs[0].set_ylabel("|D| (%)")
axs[0].set_title("Amplitud vs parámetros del rechazo")
axs[0].legend(fontsize=8, title="FWHM", title_fontsize=8); axs[0].grid(alpha=.3)
axs[1].axhline(0, color='k', ls=':', lw=1.5)
axs[1].set_xlabel("K (σ local)"); axs[1].set_ylabel("separación del CMB (grados)")
axs[1].set_title("Dirección"); axs[1].grid(alpha=.3)

et = [100*SIG_STAT/a_fid, 100*s_art/a_fid, 100*s_mask/a_fid, 10.0]
axs[2].barh(["estadístico", "artefactos", "máscara", "expectativa"], et,
            color=['#2e6fa8', '#c77a30', '#1b6b50', '#7a4fa3'])
for i, v in enumerate(et):
    axs[2].text(v+0.3, i, f"{v:.1f}%", va='center', fontsize=9)
axs[2].set_xlabel("contribución al error (%)")
axs[2].set_title(f"Presupuesto completo  ({coc:.2f} ± {e:.2f})")
axs[2].grid(alpha=.3, axis='x')
plt.tight_layout(); plt.savefig("barrido_artefactos.png", dpi=145)
print("\nFigura guardada: barrido_artefactos.png")
plt.show()

print("""
ESTE ES EL ULTIMO TERMINO QUE FALTABA.

Con el barrido hecho, el presupuesto de error queda cerrado para lo que se puede
hacer con estos datos:
   estadistico  - medido con simulaciones que incluyen clustering
   artefactos   - medido con este barrido
   mascara      - medido comparando region continua y escalera
   expectativa  - el unico que sigue siendo una estimacion (10%), y se cierra
                  calculando el boost Doppler numericamente en vez de usar la
                  formula linealizada de Ellis-Baldwin

A partir de aqui ya no hay mas sistematicos que medir con este catalogo. Lo que sigue
es escribir.
""")
