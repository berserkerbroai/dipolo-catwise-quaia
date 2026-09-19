"""
cobertura.py
------------
Identificar al contaminante del dipolo de CatWISE.

LO QUE SABEMOS
  * Hay un contaminante coherente, aislado con capas de magnitud disjuntas, en
    galactico (232.9, +13.1) = ecuatorial (123.9, -11.1) = ecliptico (129.3, -30.0).
  * Crece bruscamente hacia el limite de flujo -> es un efecto de profundidad.
  * No coincide con ningun eje obvio: el mas cercano es el anticentro galactico a 39 grados.
  * La densidad sube +2.6% cerca del plano ecliptico y baja -4.7% cerca de los polos.

LA HIPOTESIS
  La cobertura de WISE es BAJA cerca del plano ecliptico y ALTA cerca de los polos
  (el paper de CatWISE2020 lo dice explicitamente al hablar de COSMOS). Menos cobertura
  deberia dar MENOS fuentes, no mas. Asi que el signo observado NO se explica por
  profundidad a secas.

  Pero si por DISPERSION EN EL COLOR. El corte es W1-W2 >= 0.8 y la mayoria de las
  fuentes del cielo son mas azules que eso. Donde la cobertura es menor, los errores
  fotometricos son mayores, y mas fuentes azules cruzan el umbral por ruido.
  Sesgo de Eddington sobre el color. Eso predice el signo correcto.

  PREDICCION: el exceso debe concentrarse en las fuentes con color APENAS por encima
  de 0.8. Las muy rojas (W1-W2 > 1.0) son AGN reales y no deberian mostrarlo.

ESTRUCTURA
  PARTE A  - se corre YA, sin bajar nada. Usa cielo_cache.npz.
             Test del color: ¿el contaminante vive pegado al umbral de 0.8?
  PARTE B  - necesita una descarga PEQUEÑA (fuentes brillantes con la columna de
             cobertura). Construye el mapa de cobertura y hace el test diferencial:
             la densidad de fuentes DEBILES debe correlacionar con la cobertura,
             la de fuentes BRILLANTES no (son detectadas en cualquier caso).

Requiere: numpy, matplotlib. PARTE B ademas: pyvo, astropy.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

CACHE       = "cielo_cache.npz"
ARCH_COV    = "catwise_cobertura.ecsv"
CACHE_COV   = "cobertura_cache.npz"
W1_BRILLANTE = 15.0        # para el mapa de cobertura y como control
B_CUT, RAD_MC = 30.0, 8.0
NB = 180                   # rejilla equiareal para mapas (sin healpy)
HACER_PARTE_B = True

V_SOBRE_C = 369.82/299792.458
CMB_L, CMB_B = 264.021, 48.253
CONT_L, CONT_B = 232.9, 13.1        # contaminante aislado por capas
LMC, SMC = (80.894, -69.756), (13.187, -72.829)

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
    lam = np.arctan2(np.sin(ra)*np.cos(EPS) + np.tan(dec)*np.sin(EPS), np.cos(ra))
    return np.degrees(lam) % 360, np.degrees(beta)

def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], axis=-1)

def vec2lb(v):
    n = v/np.linalg.norm(v)
    return np.degrees(np.arctan2(n[1], n[0])) % 360, np.degrees(np.arcsin(np.clip(n[2], -1, 1)))

def sep_deg(l1, b1, l2, b2):
    return np.degrees(np.arccos(np.clip(np.sum(unit(l1, b1)*unit(l2, b2), axis=-1), -1, 1)))

def fit_dipolo(u_d, u_r, w_r=None):
    n = len(u_d); mu_d = u_d.mean(0)
    cov_mu = ((u_d.T @ u_d)/n - np.outer(mu_d, mu_d))/n
    if w_r is None:
        mu_r = u_r.mean(0); M = (u_r.T @ u_r)/len(u_r) - np.outer(mu_r, mu_r)
    else:
        W = w_r.sum(); mu_r = (w_r[:, None]*u_r).sum(0)/W
        M = (w_r[:, None]*u_r).T @ u_r/W - np.outer(mu_r, mu_r)
    Minv = np.linalg.inv(M); D = Minv @ (mu_d - mu_r)
    covD = Minv @ cov_mu @ Minv.T
    a = np.linalg.norm(D); nh = D/a
    return D, a, np.sqrt(max(nh @ covD @ nh, 0.0))

# =============================== CARGA ===============================
print("=" * 76)
if not os.path.exists(CACHE):
    raise SystemExit(f"Falta {CACHE}.")
z = np.load(CACHE)
ra, dec, w1, w2 = z['ra'], z['dec'], z['w1'], z['w2']
gl, gb = eq2gal(ra, dec)
_, eb = eq2ecl(ra, dec)
color = w1 - w2
print(f"PARTE A: {len(ra)} fuentes de {CACHE} (sin descarga)")

base = (np.abs(gb) > B_CUT) & (sep_deg(ra, dec, *LMC) > RAD_MC) & (sep_deg(ra, dec, *SMC) > RAD_MC)
rng = np.random.default_rng(1)
NR = 4_000_000
dr = np.degrees(np.arcsin(rng.uniform(-1, 1, NR))); rr = rng.uniform(0, 360, NR)
glr, gbr = eq2gal(rr, dr); _, ebr = eq2ecl(rr, dr)
base_r = (np.abs(gbr) > B_CUT) & (sep_deg(rr, dr, *LMC) > RAD_MC) & (sep_deg(rr, dr, *SMC) > RAD_MC)
u_r_base = unit(glr[base_r], gbr[base_r])

# =============================== PARTE A: EL COLOR ===============================
print("\n" + "=" * 76)
print("A1. GRADIENTE ECLIPTICO SEGUN CERCANIA AL UMBRAL DE COLOR")
print("    Si es dispersion fotometrica, el gradiente debe estar en las fuentes")
print("    PEGADAS a 0.8 y desaparecer en las muy rojas.\n")
bins_b = np.linspace(-90, 90, 19)
cen_b = 0.5*(bins_b[1:] + bins_b[:-1])
area, _ = np.histogram(ebr[base_r], bins=bins_b)

tramos = [(0.80, 0.90), (0.90, 1.00), (1.00, 1.20), (1.20, 1.60), (1.60, 9.99)]
print(f"    {'color W1-W2':>14} {'N':>9} {'tendencia |beta| (%)':>22} {'dens plano/polo':>17}")
grad = []
for c0, c1 in tramos:
    m = base & (color >= c0) & (color < c1) & (w1 < 16.5)
    if m.sum() < 20000:
        continue
    h, _ = np.histogram(eb[m], bins=bins_b)
    d = h/area; d /= np.average(d, weights=area)
    p = np.polyfit(np.abs(cen_b), d, 1)[0]*90*100
    plano = np.average(d[np.abs(cen_b) < 25], weights=area[np.abs(cen_b) < 25])
    polo = np.average(d[np.abs(cen_b) > 60], weights=area[np.abs(cen_b) > 60])
    print(f"    {c0:5.2f}-{c1:5.2f} {m.sum():9d} {p:+22.1f} {plano/polo:17.3f}")
    grad.append((0.5*(c0+c1), p, plano/polo, m.sum()))
grad = np.array(grad)

print("\nA2. LO MISMO PARTIDO POR MAGNITUD (la dispersion crece hacia el limite)")
print(f"    {'color':>12} {'W1<15.5':>12} {'15.5-16.0':>12} {'16.0-16.5':>12} {'16.5-17.0':>12}")
mag_tramos = [(0, 15.5), (15.5, 16.0), (16.0, 16.5), (16.5, 17.0)]
mapa_grad = np.full((len(tramos), len(mag_tramos)), np.nan)
for i, (c0, c1) in enumerate(tramos):
    fila = []
    for j, (m0, m1) in enumerate(mag_tramos):
        m = base & (color >= c0) & (color < c1) & (w1 >= m0) & (w1 < m1)
        if m.sum() < 8000:
            fila.append("      --"); continue
        h, _ = np.histogram(eb[m], bins=bins_b)
        d = h/area; d /= np.average(d, weights=area)
        pl = np.average(d[np.abs(cen_b) < 25], weights=area[np.abs(cen_b) < 25])
        po = np.average(d[np.abs(cen_b) > 60], weights=area[np.abs(cen_b) > 60])
        mapa_grad[i, j] = pl/po
        fila.append(f"{pl/po:8.3f}")
    print(f"    {c0:5.2f}-{c1:5.2f} " + " ".join(f"{x:>12}" for x in fila))
print("\n    (valor > 1 = exceso en el plano ecliptico. Si crece hacia abajo-izquierda,")
print("     o sea hacia colores pegados a 0.8 y magnitudes debiles, es dispersion.)")

print("\nA3. DIPOLO SEGUN CERCANIA AL UMBRAL")
print(f"    {'color':>12} {'N':>9} {'|D|':>17} {'l':>7} {'b':>7} {'sep CMB':>8} {'sep CONT':>9}")
dip_color = []
for c0, c1 in tramos:
    m = base & (color >= c0) & (color < c1) & (w1 < 16.5)
    if m.sum() < 20000:
        continue
    D, a, ae = fit_dipolo(unit(gl[m], gb[m]), u_r_base)
    l, b = vec2lb(D)
    print(f"    {c0:5.2f}-{c1:5.2f} {m.sum():9d} {a:.5f}+/-{ae:.5f} {l:7.1f} {b:+7.1f} "
          f"{sep_deg(l,b,CMB_L,CMB_B):8.1f} {sep_deg(l,b,CONT_L,CONT_B):9.1f}")
    dip_color.append((0.5*(c0+c1), a, ae, sep_deg(l, b, CMB_L, CMB_B),
                      sep_deg(l, b, CONT_L, CONT_B)))
dip_color = np.array(dip_color)
print("\n    Si la hipotesis es correcta: las fuentes pegadas a 0.8 apuntan al")
print("    CONTAMINANTE, y las muy rojas apuntan al CMB.")

# =============================== PARTE B: COBERTURA ===============================
if HACER_PARTE_B:
    print("\n" + "=" * 76)
    print("PARTE B: MAPA DE COBERTURA")

    def descubrir_columnas():
        import pyvo as vo
        s = vo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
        q = ("SELECT column_name, description FROM TAP_SCHEMA.columns "
             "WHERE table_name = 'catwise_2020'")
        t = s.search(q).to_table()
        nom = np.array([str(x) for x in t['column_name']])
        des = np.array([str(x) for x in t['description']])
        sel = [i for i, (n, d) in enumerate(zip(nom, des))
               if 'cov' in n.lower() or 'coverage' in d.lower()]
        print("    Columnas relacionadas con cobertura en catwise_2020:")
        for i in sel:
            print(f"      {nom[i]:<14} {des[i][:70]}")
        return [nom[i] for i in sel]

    def bajar(colcov):
        import pyvo as vo
        from astropy.table import Table, vstack
        s = vo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
        tt = []
        for nm, cond in [("N1", "dec >= 0 AND ra < 180"), ("N2", "dec >= 0 AND ra >= 180"),
                         ("S1", "dec < 0 AND ra < 180"),  ("S2", "dec < 0 AND ra >= 180")]:
            q = f"""SELECT ra, dec, w1mpro, {colcov}
                    FROM catwise_2020
                    WHERE w1mpro - w2mpro >= 0.8 AND w1snr >= 5 AND w2snr >= 5
                      AND w1mpro < {W1_BRILLANTE} AND {cond}"""
            print(f"      [{nm}] en cola...")
            j = s.submit_job(q); j.execution_duration = 3600; j.run()
            j.wait(phases=['COMPLETED', 'ERROR', 'ABORTED'])
            if j.phase != 'COMPLETED':
                j.raise_if_error(); raise RuntimeError(j.phase)
            tt.append(j.fetch_result().to_table())
            print(f"      [{nm}] {len(tt[-1])} filas")
        t = vstack(tt); t.write(ARCH_COV, overwrite=True); return t

    if os.path.exists(CACHE_COV):
        zz = np.load(CACHE_COV)
        ra_c, dec_c, cov_c = zz['ra'], zz['dec'], zz['cov']
        print(f"    cache: {len(ra_c)} fuentes brillantes con cobertura")
    else:
        from astropy.table import Table
        if os.path.exists(ARCH_COV):
            t = Table.read(ARCH_COV)
        else:
            cands = descubrir_columnas()
            colcov = next((c for c in cands if c.lower() in ('w1cov', 'w1_cov')), None) \
                     or (cands[0] if cands else None)
            if colcov is None:
                raise SystemExit("    No hay columna de cobertura; revisa la lista de arriba.")
            print(f"    usando la columna '{colcov}'. Descarga de fuentes W1<{W1_BRILLANTE}")
            print("    (es pequeña: la cobertura es propiedad del cielo, no hace falta ir profundo)")
            t = bajar(colcov)
        cc = {c.lower(): c for c in t.colnames}
        colc = next(cc[k] for k in cc if 'cov' in k)
        ra_c = np.asarray(t['ra'], float); dec_c = np.asarray(t['dec'], float)
        cov_c = np.asarray(t[colc], float)
        np.savez_compressed(CACHE_COV, ra=ra_c, dec=dec_c, cov=cov_c)
        print(f"    {len(ra_c)} fuentes brillantes, cobertura {np.nanmin(cov_c):.0f}"
              f"-{np.nanmax(cov_c):.0f}")

    # mapa de cobertura en rejilla equiareal (sin healpy)
    def celda(r, d):
        i = np.clip(((np.sin(np.radians(d)) + 1)/2*NB).astype(int), 0, NB-1)
        j = np.clip((r/360*NB).astype(int), 0, NB-1)
        return i*NB + j
    cid_c = celda(ra_c, dec_c)
    suma = np.bincount(cid_c, weights=np.nan_to_num(cov_c), minlength=NB*NB)
    cuenta = np.bincount(cid_c, minlength=NB*NB).astype(float)
    mapa_cov = np.where(cuenta > 0, suma/np.maximum(cuenta, 1), np.nan)

    # centros de celda
    ii, jj = np.divmod(np.arange(NB*NB), NB)
    dec_pix = np.degrees(np.arcsin((ii + 0.5)/NB*2 - 1))
    ra_pix = (jj + 0.5)/NB*360
    l_pix, b_pix = eq2gal(ra_pix, dec_pix)
    _, be_pix = eq2ecl(ra_pix, dec_pix)
    ok_pix = np.isfinite(mapa_cov) & (np.abs(b_pix) > B_CUT) \
             & (sep_deg(ra_pix, dec_pix, *LMC) > RAD_MC) \
             & (sep_deg(ra_pix, dec_pix, *SMC) > RAD_MC)

    print(f"\nB1. ¿LA COBERTURA SIGUE LA ECLIPTICA?")
    for lo in (0, 20, 40, 60, 80):
        s = ok_pix & (np.abs(be_pix) >= lo) & (np.abs(be_pix) < lo + 20)
        if s.sum() > 20:
            print(f"    |beta| {lo:2d}-{lo+20:2d}: cobertura media = {np.nanmean(mapa_cov[s]):7.1f}")

    print(f"\nB2. DENSIDAD vs COBERTURA  (test diferencial)")
    print("    Las BRILLANTES se detectan pase lo que pase -> deben salir planas.")
    print("    Las DEBILES son las que dependen de la profundidad.\n")
    cid_all = celda(ra, dec)
    qs = np.nanquantile(mapa_cov[ok_pix], np.linspace(0, 1, 7))
    print(f"    {'cobertura':>16} {'n celdas':>9} {'dens debil':>12} {'dens brillante':>15}")
    cov_mid, dens_faint, dens_bright = [], [], []
    for k in range(len(qs)-1):
        s = ok_pix & (mapa_cov >= qs[k]) & (mapa_cov < qs[k+1])
        if s.sum() < 20:
            continue
        cels = np.where(s)[0]
        deb = base & (w1 > 16.0) & (w1 < 17.0) & np.isin(cid_all, cels)
        bri = base & (w1 < 15.5) & np.isin(cid_all, cels)
        df, db = deb.sum()/s.sum(), bri.sum()/s.sum()
        cov_mid.append(0.5*(qs[k]+qs[k+1])); dens_faint.append(df); dens_bright.append(db)
        print(f"    {qs[k]:7.1f}-{qs[k+1]:6.1f} {s.sum():9d} {df:12.2f} {db:15.2f}")
    cov_mid = np.array(cov_mid)
    dens_faint = np.array(dens_faint)/np.mean(dens_faint)
    dens_bright = np.array(dens_bright)/np.mean(dens_bright)
    print(f"\n    variacion debiles   : {100*(dens_faint.max()-dens_faint.min()):.1f}%")
    print(f"    variacion brillantes: {100*(dens_bright.max()-dens_bright.min()):.1f}%")
    print("    Si las debiles varian MUCHO mas, la cobertura es el mecanismo.")

    print(f"\nB3. DIPOLO DEL PROPIO MAPA DE COBERTURA")
    u_pix = unit(l_pix[ok_pix], b_pix[ok_pix])
    cv = mapa_cov[ok_pix]
    mu = (cv[:, None]*u_pix).sum(0)/cv.sum() - u_pix.mean(0)
    Mm = (u_pix.T @ u_pix)/len(u_pix) - np.outer(u_pix.mean(0), u_pix.mean(0))
    Dcov = np.linalg.solve(Mm, mu)
    lc, bc = vec2lb(Dcov)
    print(f"    direccion del gradiente de cobertura: (l,b) = ({lc:.1f}, {bc:+.1f})")
    print(f"       a {sep_deg(lc,bc,CONT_L,CONT_B):.1f} grados del CONTAMINANTE ({CONT_L},{CONT_B:+.1f})")
    print(f"       a {sep_deg(lc,bc,CMB_L,CMB_B):.1f} grados del dipolo CMB")
    print("\n    SI ESTA CERCA DEL CONTAMINANTE: caso cerrado, es la cobertura.")
    print("    SI NO: la cobertura no es el culpable y hay que seguir buscando.")

# =============================== FIGURA ===============================
n = 3 if HACER_PARTE_B else 2
fig, axs = plt.subplots(1, n, figsize=(5.2*n, 4.4), dpi=145)

axs[0].plot(grad[:, 0], grad[:, 2], 'o-', color='#b5423a', lw=2, ms=8)
axs[0].axhline(1, color='k', ls=':', lw=1.5)
axs[0].set_xlabel("color W1 - W2"); axs[0].set_ylabel("densidad plano / polo eclíptico")
axs[0].set_title("¿El gradiente vive pegado al corte?"); axs[0].grid(alpha=.3)

axs[1].errorbar(dip_color[:, 0], dip_color[:, 1]*100, yerr=dip_color[:, 2]*100,
                fmt='o-', color='#2e6fa8', lw=2, ms=7, capsize=3, label='|D|')
ax1b = axs[1].twinx()
ax1b.plot(dip_color[:, 0], dip_color[:, 3], 's--', color='#c77a30', lw=1.8, ms=6,
          label='sep del CMB')
ax1b.plot(dip_color[:, 0], dip_color[:, 4], '^--', color='#1b6b50', lw=1.8, ms=6,
          label='sep del contaminante')
ax1b.set_ylabel("separación (grados)")
axs[1].set_xlabel("color W1 - W2"); axs[1].set_ylabel("|D| (%)")
axs[1].set_title("Dipolo según cercanía al umbral")
h1, l1 = axs[1].get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
axs[1].legend(h1+h2, l1+l2, fontsize=8); axs[1].grid(alpha=.3)

if HACER_PARTE_B:
    axs[2].plot(cov_mid, dens_faint, 'o-', color='#b5423a', lw=2, ms=7, label='W1 16.0-17.0 (débiles)')
    axs[2].plot(cov_mid, dens_bright, 's--', color='0.4', lw=2, ms=6, label='W1 < 15.5 (control)')
    axs[2].axhline(1, color='k', ls=':', lw=1.2)
    axs[2].set_xlabel("cobertura media"); axs[2].set_ylabel("densidad relativa")
    axs[2].set_title("Test diferencial de profundidad")
    axs[2].legend(fontsize=8.5); axs[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig("cobertura.png", dpi=145)
print("\nFigura guardada: cobertura.png")
plt.show()
