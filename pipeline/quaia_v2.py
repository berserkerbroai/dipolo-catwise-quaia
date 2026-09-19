"""
quaia_v2.py
-----------
El control fallo: la muestra completa dio cociente 4.88 con la direccion a 53 grados
del CMB, cuando lo publicado para Quaia es D ~ 1.1e-2 apuntando CERCA del CMB, con un
corte galactico ESTRICTO de |b|>40. La expectativa cinematica publicada para Quaia
G<20.0 es D ~ 0.0080, no los 0.00503 que salieron.

Este script hace el test que decide: recorre el corte galactico y mira si el dipolo
(y la supuesta evolucion con z) sobrevive cuando la Via Lactea sale de la muestra.

CAMBIOS RESPECTO A quaia_dipolo_z.py

  1. BUCLE DE MASCARA. |b| in {15,25,30,40,50}. Es el mismo test de estabilidad que
     ya delato a CatWISE. Una cantidad fisica no depende del corte; una sistematica si.

  2. BORDES DE z FIJOS, no por cuantiles. Con cuantiles los bordes cambian al cambiar
     la mascara y nada es comparable entre corridas. Con bordes fijos, cada capa es la
     misma poblacion en todas las mascaras y en los dos catalogos.

  3. LOS DOS CATALOGOS. G20.0 y G20.5 se analizan en paralelo. Son muestras distintas
     con profundidades distintas: si la señal es fisica debe aparecer en ambas.
     OJO: cada uno necesita SU PROPIA funcion de seleccion. Usar la de G20.0 sobre el
     catalogo G20.5 seria un error grave.

  4. alpha MEDIDO del color BP-RP en vez de fijado a 0.61. El catalogo trae
     phot_bp_mean_mag y phot_rp_mean_mag. Un indice espectral inventado fue parte de
     por que D_esp salio demasiado pequeño.

  5. TEST DE LA PREDICCION. Si la capa de z bajo esta contaminada por estrellas del
     bulbo mal clasificadas como cuasares, debe moverse MUCHO MAS que las otras al
     endurecer el corte. El script lo cuantifica explicitamente.

ARCHIVOS
  quaia_G20.0.fits, quaia_G20.5.fits
  selection_function_NSIDE64_G20.0.fits      <- imprescindible
  selection_function_NSIDE64_G20.5.fits      <- si falta, G20.5 solo se usa para x

Requiere: numpy, matplotlib, astropy, healpy.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.table import Table

# =============================== CONFIGURACION ===============================
CATALOGOS = [
    ("G20.0", "quaia_G20.0.fits", "selection_function_NSIDE64_G20.0.fits", 20.0),
    ("G20.5", "quaia_G20.5.fits", "selection_function_NSIDE64_G20.5.fits", 20.5),
]
B_CUTS = [15.0, 25.0, 30.0, 40.0, 50.0]
SEL_MIN = 0.60
RAD_MC = 8.0
Z_BORDES = np.array([0.10, 1.50, 4.60])   # Dos capas anchas: z < 1.5 y z > 1.5
ALPHA_FIJO = 0.61                 # solo como referencia; se usa el medido del color
D_ESP_PUBLICADO = 0.0080          # expectativa cinematica publicada para Quaia G<20.0
D_PUBLICADO = 0.011           # amplitud publicada con |b|>40

V_SOBRE_C = 369.82/299792.458
CMB_L, CMB_B = 264.021, 48.253
LMC, SMC = (80.894, -69.756), (13.187, -72.829)

# Gaia DR3, sistema Vega. Si tu version difiere, cambialos aqui.
F0_BP, F0_RP = 3552.01, 2554.95          # Jy
LAM_BP, LAM_RP = 5109.7, 7769.0          # Angstrom

# =============================== COORDENADAS ===============================
RA_NGP, DEC_NGP, L_NCP = np.radians(
    192.85948), np.radians(27.12825), np.radians(122.93192)


def eq2gal(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    b = np.arcsin(np.clip(np.sin(dec)*np.sin(DEC_NGP) +
                          np.cos(dec)*np.cos(DEC_NGP)*np.cos(ra-RA_NGP), -1, 1))
    y = np.cos(dec)*np.sin(ra-RA_NGP)
    x = np.sin(dec)*np.cos(DEC_NGP) - np.cos(dec) * \
        np.sin(DEC_NGP)*np.cos(ra-RA_NGP)
    return np.degrees(L_NCP - np.arctan2(y, x)) % 360, np.degrees(b)


def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], axis=-1)


def vec2lb(v):
    n = v/np.linalg.norm(v)
    return np.degrees(np.arctan2(n[1], n[0])) % 360, np.degrees(np.arcsin(np.clip(n[2], -1, 1)))


def sep_deg(l1, b1, l2, b2):
    return np.degrees(np.arccos(np.clip(np.sum(unit(l1, b1)*unit(l2, b2), axis=-1), -1, 1)))


def color_a_alpha_gaia(bp_rp):
    """S_nu ~ nu^-alpha a partir de BP-RP (Vega).
    S_BP/S_RP = (F0_BP/F0_RP)*10^(-(BP-RP)/2.5) ; nu_BP/nu_RP = lam_RP/lam_BP."""
    ratio = (F0_BP/F0_RP)*10**(-bp_rp/2.5)
    return -np.log(ratio)/np.log(LAM_RP/LAM_BP)

# =============================== ESTIMADOR ===============================


def fit_dipolo(u_d, u_r, w_r):
    """dN/dOmega ~ w(n)*(1 + D.n). La funcion de seleccion entra como peso de los
    centros de pixel, que hacen de catalogo aleatorio exacto (sin ruido Monte Carlo)."""
    n = len(u_d)
    mu_d = u_d.mean(0)
    cov_mu = ((u_d.T @ u_d)/n - np.outer(mu_d, mu_d))/n
    Wr = w_r.sum()
    mu_r = (w_r[:, None]*u_r).sum(0)/Wr
    M = (w_r[:, None]*u_r).T @ u_r/Wr - np.outer(mu_r, mu_r)
    Minv = np.linalg.inv(M)
    D = Minv @ (mu_d - mu_r)
    covD = Minv @ cov_mu @ Minv.T
    a = np.linalg.norm(D)
    nh = D/a
    return D, a, np.sqrt(max(nh @ covD @ nh, 0.0))


def pendiente_local(mags, m0, ventana=0.35, un_lado=False):
    mags = np.sort(mags[np.isfinite(mags)])
    lo, hi = (m0 - ventana, m0) if un_lado else (m0 - ventana, m0 + ventana)
    mm = np.linspace(lo, hi, 20)
    NN = np.searchsorted(mags, mm)
    g = NN > 30
    return np.polyfit(mm[g], np.log10(NN[g]), 1)[0]/0.4 if g.sum() >= 5 else np.nan


def leer_mapa(f):
    try:
        return hp.read_map(f, verbose=False)
    except TypeError:
        return hp.read_map(f)


# =============================== CARGA ===============================
print("=" * 78)
print("1. CARGA")
datos = {}
for nombre, fcat, fsel, glim in CATALOGOS:
    if not os.path.exists(fcat):
        print(f"   {nombre}: falta {fcat}, se omite")
        continue
    t = Table.read(fcat)
    cols = {c.lower(): c for c in t.colnames}
    g = lambda *c: next((cols[x] for x in c if x in cols), None)
    c_ra, c_dec = g('ra'), g('dec')
    c_z = g('redshift_quaia', 'z', 'redshift')
    c_g = g('phot_g_mean_mag', 'mag_g_gaia')
    c_bp, c_rp = g('phot_bp_mean_mag'), g('phot_rp_mean_mag')
    ra = np.asarray(t[c_ra], float)
    dec = np.asarray(t[c_dec], float)
    zq = np.asarray(t[c_z], float)
    gm = np.asarray(t[c_g], float) if c_g else np.full(len(ra), np.nan)
    if c_bp and c_rp:
        bp_rp = np.asarray(t[c_bp], float) - np.asarray(t[c_rp], float)
    else:
        bp_rp = np.full(len(ra), np.nan)
    gl, gb = eq2gal(ra, dec)

    sel = pix_src = l_pix = b_pix = None
    if os.path.exists(fsel):
        sel = np.nan_to_num(leer_mapa(fsel), nan=0.0)
        nside = hp.npix2nside(len(sel))
        th, ph = hp.pix2ang(nside, np.arange(len(sel)))
        lat_m, lon_m = 90.0 - np.degrees(th), np.degrees(ph)
        # ¿mapa en galacticas o ecuatoriales? la seleccion debe CAER en el plano galactico
        def con(la): return sel[np.abs(la) > 50].mean() / \
            max(sel[np.abs(la) < 10].mean(), 1e-6)
        _, lat_eq = eq2gal(lon_m, lat_m)
        es_gal = con(lat_m) >= con(lat_eq)
        if es_gal:
            l_pix, b_pix = lon_m, lat_m
            pix_src = hp.ang2pix(nside, np.radians(90.0 - gb), np.radians(gl))
        else:
            l_pix, b_pix = eq2gal(lon_m, lat_m)
            pix_src = hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra))
        marco = 'GALACTICO' if es_gal else 'ECUATORIAL'
    else:
        marco = 'SIN MAPA'
    print(f"   {nombre}: {len(ra)} fuentes, Glim={glim}, "
          f"seleccion={'ok (' + marco + ')' if sel is not None else 'FALTA ' + fsel}")
    datos[nombre] = dict(ra=ra, dec=dec, z=zq, g=gm, bp_rp=bp_rp, gl=gl, gb=gb,
                         sel=sel, pix=pix_src, l_pix=l_pix, b_pix=b_pix, glim=glim)

if not datos:
    raise SystemExit("No hay catalogos.")
if 'G20.0' in datos and datos['G20.0']['sel'] is None:
    print("\n   *** Sin la funcion de seleccion de G20.0 no tiene sentido continuar.")
    print("   Archivos .fits presentes:", glob.glob("*.fits"))
    raise SystemExit()

# =============================== x y alpha ===============================
print("\n2. PENDIENTE DE CONTEOS x  Y  INDICE ESPECTRAL alpha")


def x_en(nombre, m0, zmin=None, zmax=None):
    """x en m0. Para G20.0 se mide sobre G20.5 (sin truncamiento). Para G20.5 no hay
    nada mas profundo, asi que el ajuste es de un solo lado y queda sesgado a la baja."""
    fuente = 'G20.5' if (nombre == 'G20.0' and 'G20.5' in datos) else nombre
    d = datos[fuente]
    m = np.isfinite(d['g'])
    if zmin is not None:
        m &= (d['z'] > zmin) & (d['z'] <= zmax)
    return pendiente_local(d['g'][m], m0, un_lado=(fuente == nombre)), fuente


for nombre in datos:
    xv, src = x_en(nombre, datos[nombre]['glim'])
    d = datos[nombre]
    ok = np.isfinite(d['bp_rp'])
    al = np.nanmedian(color_a_alpha_gaia(
        d['bp_rp'][ok])) if ok.any() else ALPHA_FIJO
    d['x_tot'], d['alpha_tot'] = xv, al
    print(f"   {nombre}: x={xv:.3f} (medido en {src}{' , UN LADO -> sesgado' if src == nombre else ''})"
          f"   alpha(BP-RP)={al:.3f}   [alpha fijo de referencia={ALPHA_FIJO}]")
    d['D_esp_tot'] = (2 + xv*(1 + al))*V_SOBRE_C
    print(f"           D_esp = {d['D_esp_tot']:.5f}   "
          f"[publicado para G<20.0: ~{D_ESP_PUBLICADO:.4f}]")

# =============================== MASCARA + AJUSTE ===============================


def analizar(nombre, bcut):
    d = datos[nombre]
    if d['sel'] is None:
        return None
    ok_pix = (d['sel'] > SEL_MIN) & (np.abs(d['b_pix']) > bcut)
    for c in (LMC, SMC):
        cl, cb = eq2gal(*c)
        ok_pix &= sep_deg(d['l_pix'], d['b_pix'], cl, cb) > RAD_MC
    u_r, w_r = unit(d['l_pix'][ok_pix], d['b_pix'][ok_pix]), d['sel'][ok_pix]
    ok_src = ok_pix[d['pix']] & np.isfinite(d['z'])
    u_all = unit(d['gl'][ok_src], d['gb'][ok_src])
    return dict(u_r=u_r, w_r=w_r, ok=ok_src, u=u_all, z=d['z'][ok_src],
                f_sky=ok_pix.sum()/len(d['sel']))


print("\n" + "=" * 78)
print("3. CONTROL: MUESTRA COMPLETA vs CORTE GALACTICO")
print(
    f"   (referencia publicada: |D|~{D_PUBLICADO:.4f} con |b|>40, cerca de la direccion del CMB)")
control = {}
for nombre in datos:
    if datos[nombre]['sel'] is None:
        print(f"\n   {nombre}: sin funcion de seleccion, se omite el dipolo")
        continue
    print(f"\n   --- {nombre} ---")
    print(f"   {'|b|>':>5} {'f_sky':>6} {'N':>8} {'|D|':>17} {'l':>7} {'b':>7} {'sep':>6} {'coc':>6}")
    fila = []
    for bc in B_CUTS:
        A = analizar(nombre, bc)
        D, a, ae = fit_dipolo(A['u'], A['u_r'], A['w_r'])
        l, b = vec2lb(D)
        s = sep_deg(l, b, CMB_L, CMB_B)
        coc = a/datos[nombre]['D_esp_tot']
        print(f"   {bc:5.0f} {A['f_sky']:6.3f} {A['ok'].sum():8d} {a:.5f}+/-{ae:.5f} "
              f"{l:7.1f} {b:+7.1f} {s:6.1f} {coc:6.2f}")
        fila.append((bc, A['f_sky'], A['ok'].sum(), a, ae, l, b, s, coc))
    control[nombre] = np.array(fila, dtype=float)

# =============================== CAPAS DE z vs MASCARA ===============================
print("\n" + "=" * 78)
print("4. CAPAS DE z, PARA CADA CORTE GALACTICO")
print(f"   bordes FIJOS: {Z_BORDES}")
zres = {}
for nombre in datos:
    if datos[nombre]['sel'] is None:
        continue
    print(f"\n   ===== {nombre} =====")
    tabla = []
    for bc in B_CUTS:
        A = analizar(nombre, bc)
        print(
            f"\n   |b|>{bc:.0f}   (N={A['ok'].sum()}, f_sky={A['f_sky']:.3f})")
        print(f"      {'capa z':>12} {'N':>8} {'x':>6} {'D_esp':>8} {'|D|':>17} "
              f"{'l':>7} {'b':>7} {'sep':>6} {'coc':>12}")
        f_bc = []
        for k in range(len(Z_BORDES) - 1):
            z0, z1 = Z_BORDES[k], Z_BORDES[k+1]
            m = (A['z'] > z0) & (A['z'] <= z1)
            if m.sum() < 2000:
                continue
            D, a, ae = fit_dipolo(A['u'][m], A['u_r'], A['w_r'])
            l, b = vec2lb(D)
            s = sep_deg(l, b, CMB_L, CMB_B)
            xk, _ = x_en(nombre, datos[nombre]['glim'], z0, z1)
            De = (2 + xk*(1 + datos[nombre]['alpha_tot']))*V_SOBRE_C
            print(f"      {z0:5.2f}-{z1:5.2f} {m.sum():8d} {xk:6.3f} {De:8.5f} "
                  f"{a:.5f}+/-{ae:.5f} {l:7.1f} {b:+7.1f} {s:6.1f} "
                  f"{a/De:5.2f}+/-{ae/De:4.2f}")
            f_bc.append((bc, 0.5*(z0+z1), m.sum(), xk,
                        De, a, ae, l, b, s, a/De, ae/De))
        f = np.array(f_bc, dtype=float)
        if len(f) > 2:
            w = 1/f[:, 11]**2
            cm = np.sum(f[:, 10]*w)/np.sum(w)
            chi = np.sum((f[:, 10]-cm)**2*w)
            dof = len(f)-1
            print(f"      -> cociente medio {cm:.2f} ; chi2 contra constante "
                  f"{chi:.2f}/{dof} = {chi/dof:.2f}")
        tabla.append(f)
    zres[nombre] = np.vstack(tabla)

# =============================== TEST DE LA PREDICCION ===============================
print("\n" + "=" * 78)
print("5. TEST DE LA PREDICCION")
print("   Si la capa de z bajo esta contaminada por estrellas del bulbo mal")
print("   clasificadas, debe moverse MUCHO MAS que las otras al endurecer el corte.")
for nombre, R in zres.items():
    print(f"\n   --- {nombre}: cambio entre |b|>{B_CUTS[0]:.0f} y |b|>40 ---")
    r0 = R[R[:, 0] == B_CUTS[0]]
    r1 = R[R[:, 0] == 40.0]
    if len(r0) == 0 or len(r1) == 0:
        continue
    print(f"      {'capa z':>8} {'|D| suelto':>11} {'|D| estricto':>13} "
          f"{'cambio':>9} {'sep suelto':>11} {'sep estricto':>13}")
    for zc in np.unique(r0[:, 1]):
        a0 = r0[r0[:, 1] == zc]
        a1 = r1[r1[:, 1] == zc]
        if len(a0) == 0 or len(a1) == 0:
            continue
        d0, d1 = a0[0, 5], a1[0, 5]
        print(f"      z~{zc:5.2f} {d0:11.5f} {d1:13.5f} {100*(d1-d0)/d0:+8.1f}% "
              f"{a0[0, 9]:11.1f} {a1[0, 9]:13.1f}")

# =============================== FIGURA ===============================
n_cat = len(zres)
fig, axs = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
col = plt.cm.plasma(np.linspace(0, .8, len(B_CUTS)))
est = {'G20.0': '-o', 'G20.5': '--s'}

for nombre, C in control.items():
    axs[0, 0].errorbar(C[:, 0], C[:, 3]*100, yerr=C[:, 4]*100, fmt=est.get(nombre, '-o'),
                       lw=2, ms=6, capsize=3, label=nombre)
axs[0, 0].axhline(D_PUBLICADO*100, color='crimson', ls=':', lw=2,
                  label=f'publicado (|b|>40): {D_PUBLICADO*100:.1f}%')
axs[0, 0].set_xlabel("corte |b| (grados)")
axs[0, 0].set_ylabel("|D| (%)")
axs[0, 0].set_title("Control: ¿la amplitud depende de la máscara?")
axs[0, 0].legend(fontsize=8.5)
axs[0, 0].grid(alpha=.3)

for nombre, C in control.items():
    axs[0, 1].plot(C[:, 0], C[:, 7], est.get(
        nombre, '-o'), lw=2, ms=6, label=nombre)
axs[0, 1].axhline(0, color='k', ls=':', lw=1.5, label='dirección del CMB')
axs[0, 1].set_xlabel("corte |b| (grados)")
axs[0, 1].set_ylabel("separación del CMB (grados)")
axs[0, 1].set_title("Control: ¿apunta al CMB?")
axs[0, 1].legend(fontsize=8.5)
axs[0, 1].grid(alpha=.3)

R = zres.get('G20.0')
if R is not None:
    for i, bc in enumerate(B_CUTS):
        s = R[R[:, 0] == bc]
        if len(s) == 0:
            continue
        axs[1, 0].errorbar(s[:, 1], s[:, 10], yerr=s[:, 11], fmt='-o', color=col[i],
                           lw=1.8, ms=6, capsize=3, label=f"|b|>{bc:.0f}")
        axs[1, 1].plot(s[:, 1], s[:, 9], '-o', color=col[i],
                       lw=1.8, ms=6, label=f"|b|>{bc:.0f}")
axs[1, 0].axhline(1, color='k', ls=':', lw=1.5)
axs[1, 0].set_xlabel("z")
axs[1, 0].set_ylabel(r"$D_{obs}/D_{esp}$")
axs[1, 0].set_title(
    "G20.0: ¿sobrevive la evolución con z al endurecer la máscara?")
axs[1, 0].legend(fontsize=8)
axs[1, 0].grid(alpha=.3)
axs[1, 1].axhline(0, color='k', ls=':', lw=1.5)
axs[1, 1].set_xlabel("z")
axs[1, 1].set_ylabel("separación del CMB (grados)")
axs[1, 1].set_title("G20.0: dirección por capa de z")
axs[1, 1].legend(fontsize=8)
axs[1, 1].grid(alpha=.3)

plt.tight_layout()
plt.savefig("quaia_v2.png", dpi=140)
print("\nFigura guardada: quaia_v2.png")
plt.show()

print("""
COMO LEERLO

 * Panel superior izquierdo. Si |D| cae al endurecer |b| y se acerca a la linea roja,
   lo que medias antes era la Via Lactea. Si se queda plano, la mascara no era el
   problema y hay que buscar en otra parte.
 * Panel superior derecho. La separacion del CMB deberia ACERCARSE a cero al limpiar.
   Lo publicado con |b|>40 apunta cerca del CMB.
 * Paneles inferiores. La pregunta real: las curvas de distintos |b| deberian
   superponerse. Si el chi2 de evolucion con z solo es grande con mascaras sueltas y
   desaparece con |b|>40, la 'evolucion' era la galaxia.
 * Los dos catalogos deben coincidir dentro de sus barras. G20.5 tiene mas fuentes
   pero mas contaminacion; si difieren mucho, gana el mas limpio.

LO QUE SIGUE SIN ESTAR RESUELTO

 * x para G20.5 se mide de un solo lado (no hay catalogo mas profundo) y esta sesgado
   a la baja, asi que su D_esp es un limite inferior y su cociente un limite superior.
 * alpha del color BP-RP se ve afectado por el enrojecimiento galactico: en mascaras
   sueltas queda sesgado. Es otra razon para fiarte mas de |b|>40.
 * Las barras siguen siendo de muestreo, sin clustering.
""")
