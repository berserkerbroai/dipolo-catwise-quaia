"""
clustering.py
-------------
Barras de error que incluyen el clustering de cuasares, y la significancia real.

EL PROBLEMA
  Todas las barras calculadas hasta ahora son de muestreo: suponen que las fuentes
  caen de forma independiente, proceso de Poisson puro. No es cierto. Los cuasares
  trazan la estructura a gran escala y estan agrupados, lo que anade varianza al
  dipolo medido. Con la barra equivocada, el cociente 2.07 +/- 0.30 no se puede
  convertir en sigmas.

LA SOLUCION
  Catalogos simulados. Se mide el espectro de potencias angular C_l de los datos, se
  generan realizaciones con ese mismo C_l, se les inyecta el dipolo CINEMATICO
  esperado, y se pasan por la MISMA mascara y el MISMO estimador. La distribucion de
  amplitudes recuperadas es la hipotesis nula: "el dipolo es puramente cinematico".
  La fraccion de simulaciones que superan el valor observado es el p-valor.

ESTRUCTURA
  1. Mapa de densidad y estimador basado en pixeles (verificado contra el de puntos)
  2. Medicion de C_l con correccion por f_sky y resta de ruido de disparo
  3. Simulaciones NULAS (sin dipolo): valida que el estimador no sesga
  4. Test de recuperacion: se inyectan dipolos conocidos y se comprueba que salen
  5. Simulaciones CINEMATICAS: la hipotesis nula real
  6. p-valor y significancia del exceso

TODO SOBRE HEALPix, para que mascara, C_l y simulaciones vivan en la misma rejilla.
El rechazo de celdas se rehace aqui sobre pixeles HEALPix por consistencia.

Requiere: numpy, matplotlib, healpy, scipy.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy import stats

# =============================== CONFIGURACION ===============================
CACHE = "cielo_cache.npz"
NSIDE = 64
B_CUT, RAD_MC, K_ART = 30.0, 8.0, 12.0

# La muestra final del estudio
COLOR_MIN, COLOR_MAX, W1_MAX = 0.80, 1.60, 16.00
D_ESPERADO = 0.00670          # expectativa cinematica para esta muestra
N_MOCKS    = 2000
LMAX_FIT   = 3*NSIDE - 1

V_SOBRE_C = 369.82/299792.458
CMB_L, CMB_B = 264.021, 48.253
LMC, SMC = (80.894, -69.756), (13.187, -72.829)

# =============================== COORDENADAS ===============================
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

# =============================== 1. MAPA Y ESTIMADOR ===============================
print("=" * 74)
print("1. MAPA DE DENSIDAD Y MASCARA")
z = np.load(CACHE)
ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
gl, gb = eq2gal(ra, dec); color = w1 - w2

sel = ((np.abs(gb) > B_CUT) & (color >= COLOR_MIN) & (color < COLOR_MAX) & (w1 < W1_MAX)
       & (sep_deg(ra, dec, *LMC) > RAD_MC) & (sep_deg(ra, dec, *SMC) > RAD_MC))
NPIX = hp.nside2npix(NSIDE)
pix = hp.ang2pix(NSIDE, np.radians(90.0 - gb[sel]), np.radians(gl[sel]))
conteo = np.bincount(pix, minlength=NPIX).astype(float)

# geometria de los pixeles, en galacticas
th, ph = hp.pix2ang(NSIDE, np.arange(NPIX))
l_pix, b_pix = np.degrees(ph), 90.0 - np.degrees(th)
u_pix = unit(l_pix, b_pix)

# mascara base: los pixeles cuyo centro cumple los cortes geometricos
ra_pix, dec_pix = None, None
# el corte |b| y las Nubes se evaluan en galacticas / via separacion angular
lmc_l, lmc_b = eq2gal(*LMC); smc_l, smc_b = eq2gal(*SMC)
mask = ((np.abs(b_pix) > B_CUT)
        & (sep_deg(l_pix, b_pix, lmc_l, lmc_b) > RAD_MC)
        & (sep_deg(l_pix, b_pix, smc_l, smc_b) > RAD_MC))

# rechazo de artefactos, rehecho sobre HEALPix
ocup = conteo[mask & (conteo > 0)]
mu_pix = ocup.mean()
malas = conteo > mu_pix + K_ART*np.sqrt(mu_pix)
mask &= ~malas
f_sky = mask.mean()
n_tot = conteo[mask].sum()
n_bar = n_tot/mask.sum()                    # fuentes por pixel
print(f"   N = {int(n_tot)} fuentes en {mask.sum()} pixeles  (f_sky = {f_sky:.3f})")
print(f"   {malas.sum()} pixeles rechazados por artefactos (umbral {mu_pix:.1f} + {K_ART}*sqrt)")
print(f"   densidad media = {n_bar:.1f} fuentes/pixel")

def dipolo_de_conteos(c, m):
    """Estimador lineal sobre conteos por pixel. c = conteos, m = mascara booleana."""
    uu = u_pix[m]; cc = c[m]
    mu_d = (cc[:, None]*uu).sum(0)/cc.sum()
    mu_r = uu.mean(0)
    M = (uu.T @ uu)/len(uu) - np.outer(mu_r, mu_r)
    return np.linalg.solve(M, mu_d - mu_r)

D_obs = dipolo_de_conteos(conteo, mask)
a_obs = np.linalg.norm(D_obs); l_obs, b_obs = vec2lb(D_obs)
n_cmb = unit(CMB_L, CMB_B)
par_obs = D_obs @ n_cmb
print(f"\n   DIPOLO OBSERVADO: |D| = {a_obs:.5f}   (l,b) = ({l_obs:.1f}, {b_obs:+.1f})")
print(f"   separacion del CMB = {sep_deg(l_obs,b_obs,CMB_L,CMB_B):.1f} grados")
print(f"   componente paralela al CMB = {par_obs:.5f}")
print(f"   (comparar con el valor del estimador de puntos: ~0.0139)")

# =============================== 2. ESPECTRO DE POTENCIAS ===============================
print("\n" + "=" * 74)
print("2. ESPECTRO DE POTENCIAS ANGULAR")
delta = np.zeros(NPIX)
delta[mask] = conteo[mask]/n_bar - 1.0
cl_pseudo = hp.anafast(delta*mask, lmax=LMAX_FIT)
cl_obs = cl_pseudo/f_sky                                  # correccion cruda por f_sky
omega_pix = 4*np.pi/NPIX
shot = omega_pix/n_bar                                    # ruido de disparo, en sr
cl_sig = np.maximum(cl_obs - shot, 0.0)
cl_sig[0:2] = 0.0                                         # sin monopolo ni dipolo
print(f"   ruido de disparo = {shot:.3e} sr")
print(f"   C_l de senal:  l=2: {cl_sig[2]:.3e}   l=5: {cl_sig[5]:.3e}   "
      f"l=10: {cl_sig[10]:.3e}   l=30: {cl_sig[30]:.3e}")
print(f"   amplitud rms de las fluctuaciones de clustering: "
      f"{np.sqrt(np.sum((2*np.arange(len(cl_sig))+1)*cl_sig)/(4*np.pi)):.4f}")
print("\n   NOTA: este C_l incluye TODA la potencia del mapa, clustering real mas")
print("   sistematicas residuales. Eso hace la barra de error CONSERVADORA, que es")
print("   lo que se quiere: ninguna estructura observada queda fuera del modelo de ruido.")

# =============================== 3. MOTOR DE SIMULACIONES ===============================
rng = np.random.default_rng(42)

def simular(n_sims, dipolo_inyectado=None, semilla=0):
    """Genera n_sims mapas con el C_l medido, opcionalmente con un dipolo inyectado,
    los muestrea con Poisson y devuelve los dipolos recuperados."""
    r = np.random.default_rng(semilla)
    out = np.empty((n_sims, 3))
    for i in range(n_sims):
        d = hp.synfast(cl_sig, NSIDE, lmax=LMAX_FIT, verbose=False) \
            if 'verbose' in hp.synfast.__code__.co_varnames else \
            hp.synfast(cl_sig, NSIDE, lmax=LMAX_FIT)
        if dipolo_inyectado is not None:
            d = d + u_pix @ dipolo_inyectado
        lam = n_bar*(1.0 + d)
        lam[lam < 0] = 0.0
        c = r.poisson(lam).astype(float)
        out[i] = dipolo_de_conteos(c, mask)
    return out

# =============================== 4. NULAS Y RECUPERACION ===============================
print("\n" + "=" * 74)
print("3. SIMULACIONES NULAS  (sin dipolo inyectado)")
D_null = simular(500, None, semilla=1)
a_null = np.linalg.norm(D_null, axis=1)
print(f"   amplitud recuperada: mediana {np.median(a_null):.5f}, "
      f"percentil 95 {np.percentile(a_null, 95):.5f}")
print(f"   media vectorial: {np.linalg.norm(D_null.mean(0)):.6f}  (deberia ser ~0)")
print(f"   sigma por componente: {D_null.std(0).mean():.5f}")
sigma_clust = D_null.std(0).mean()
print(f"\n   COMPARACION DE BARRAS DE ERROR")
sigma_poisson = np.sqrt(3.0/n_tot)     # error por componente de un Poisson puro
print(f"      solo Poisson  : {sigma_poisson:.5f} por componente")
print(f"      con clustering: {sigma_clust:.5f} por componente")
print(f"      factor de inflacion: {sigma_clust/sigma_poisson:.2f}x")

print("\n4. TEST DE RECUPERACION  (se inyectan dipolos conocidos)")
print(f"   {'inyectado':>10} {'recuperado':>22} {'sesgo':>9}")
for amp in (0.005, 0.010, 0.015):
    Din = amp*n_cmb
    Dr = simular(200, Din, semilla=int(amp*1e5))
    rec = Dr @ n_cmb
    print(f"   {amp:10.4f} {rec.mean():10.5f} +/- {rec.std():8.5f} "
          f"{100*(rec.mean()-amp)/amp:+8.1f}%")

# =============================== 5. LA HIPOTESIS NULA ===============================
print("\n" + "=" * 74)
print(f"5. HIPOTESIS NULA: el dipolo es PURAMENTE CINEMATICO (D = {D_ESPERADO:.5f})")
print(f"   {N_MOCKS} simulaciones con clustering + dipolo cinematico...")
D_kin = simular(N_MOCKS, D_ESPERADO*n_cmb, semilla=2026)
a_kin = np.linalg.norm(D_kin, axis=1)
par_kin = D_kin @ n_cmb

n_exceden = (a_kin >= a_obs).sum()
p_emp = max(n_exceden, 1)/N_MOCKS
print(f"\n   distribucion de |D| bajo la nula: media {a_kin.mean():.5f}, "
      f"sigma {a_kin.std():.5f}")
print(f"   valor observado: {a_obs:.5f}")
print(f"   simulaciones que lo igualan o superan: {n_exceden} de {N_MOCKS}")

sig_gauss = (a_obs - a_kin.mean())/a_kin.std()
p_gauss = stats.norm.sf(sig_gauss)
print(f"\n   SIGNIFICANCIA")
print(f"      empirica : p {'<' if n_exceden == 0 else '='} {p_emp:.4f}  "
      f"-> {stats.norm.isf(p_emp):.2f} sigma"
      f"{'  (limitada por el numero de simulaciones)' if n_exceden == 0 else ''}")
print(f"      gaussiana: {sig_gauss:.2f} sigma  (p = {p_gauss:.2e})")
print(f"      [Secrest et al. 2021 reportan 4.9 sigma; la reevaluacion de 2025, 3.3 sigma]")

# lo mismo sobre la componente paralela, que es la cantidad fisica limpia
n_exc_par = (par_kin >= par_obs).sum()
sig_par = (par_obs - par_kin.mean())/par_kin.std()
print(f"\n   Sobre la componente PARALELA al CMB (mas limpia, sin el contaminante perp.):")
print(f"      observada {par_obs:.5f} contra {par_kin.mean():.5f} +/- {par_kin.std():.5f}")
print(f"      {n_exc_par} de {N_MOCKS} la superan  ->  {sig_par:.2f} sigma")

# el cociente, con su barra correcta
coc = a_obs/D_ESPERADO
err_coc = a_kin.std()/D_ESPERADO
print(f"\n   COCIENTE CON LA BARRA CORRECTA: {coc:.2f} +/- {err_coc:.2f}")
print(f"   (con la barra de solo muestreo era +/- 0.30)")

# =============================== 6. FIGURA ===============================
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5), dpi=145)

ll = np.arange(2, min(64, len(cl_sig)))
axs[0].semilogy(ll, cl_obs[ll], 'o-', color='0.5', ms=3, lw=1, label='medido (con ruido)')
axs[0].semilogy(ll, np.maximum(cl_sig[ll], 1e-12), 'o-', color='#b5423a', ms=3, lw=1.5,
                label='señal (ruido restado)')
axs[0].axhline(shot, color='k', ls=':', lw=1.5, label='ruido de disparo')
axs[0].set_xlabel("multipolo ℓ"); axs[0].set_ylabel(r"$C_\ell$ (sr)")
axs[0].set_title("Espectro de potencias del mapa"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)

axs[1].hist(a_null*100, bins=40, color='0.65', alpha=.75, label='nula (sin dipolo)')
axs[1].hist(a_kin*100, bins=50, color='#2e6fa8', alpha=.65, label='cinemática')
axs[1].axvline(a_obs*100, color='#b5423a', lw=2.5, label=f'observado {a_obs*100:.2f}%')
axs[1].axvline(D_ESPERADO*100, color='k', ls='--', lw=1.5, label='esperado')
axs[1].set_xlabel("|D| (%)"); axs[1].set_ylabel("simulaciones")
axs[1].set_title(f"Distribución bajo la nula  ({sig_gauss:.1f}σ)")
axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)

axs[2].hist(par_kin*100, bins=50, color='#1b6b50', alpha=.7)
axs[2].axvline(par_obs*100, color='#b5423a', lw=2.5, label=f'observado {par_obs*100:.2f}%')
axs[2].axvline(D_ESPERADO*100, color='k', ls='--', lw=1.5, label='esperado')
axs[2].set_xlabel("componente ∥ al CMB (%)"); axs[2].set_ylabel("simulaciones")
axs[2].set_title(f"Componente paralela  ({sig_par:.1f}σ)")
axs[2].legend(fontsize=8); axs[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig("clustering.png", dpi=145)
print("\nFigura guardada: clustering.png")
plt.show()

print("""
LO QUE ESTE CALCULO SI RESUELVE Y LO QUE NO

 SI. La barra de error ahora incluye la varianza del clustering, medida de tu propio
 mapa. El cociente que sale de aqui ya se puede citar con una significancia.

 NO, tres cosas:
  * El C_l se corrigio por f_sky de forma cruda. Un tratamiento riguroso de cielo
    cortado (MASTER / pseudo-C_l con deconvolucion de la matriz de acoplamiento) da
    un C_l algo distinto a multipolos bajos, que es justo donde mas pesa.
  * El C_l medido incluye sistematicas residuales, no solo clustering. Eso infla la
    barra, asi que la significancia que sale es un limite INFERIOR. Es el lado seguro.
  * El campo generado es gaussiano. El campo de densidad real no lo es a escalas
    pequenas, aunque a los multipolos que dominan el dipolo la aproximacion es buena.

 Si la significancia sale entre 3 y 4 sigma, es consistente con la reevaluacion
 reciente del campo. Si sale mucho mas alta, sospecha del C_l antes que del universo.
""")
