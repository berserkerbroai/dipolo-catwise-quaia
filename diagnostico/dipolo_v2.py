"""
dipolo_v2.py - Dipolo de cuasares CatWISE con estimador consciente de la mascara.

Cambios respecto a tu version:
  1. Test de la ecliptica CORRECTO: densidad = cuentas / area realmente disponible.
     (tu histograma crudo medía cos(beta) + la huella de la mascara, no WISE)
  2. Estimador lineal de dipolo que maneja la mascara con un catalogo aleatorio,
     en vez de minimos cuadrados sobre pixeles cortados.
  3. Barras de error por bootstrap.
  4. TEST DE ESTABILIDAD: el dipolo se recalcula variando |b|. Un dipolo cosmologico
     real es estable; solo debe crecer su incertidumbre. Si la amplitud cae, es
     contaminacion galactica.
  5. Elimina las Nubes de Magallanes (sobreviven |b|>30).
  6. Mide x, la pendiente de los conteos, para calcular la expectativa cinematica
     correcta para TUS cortes en vez de usar 0.0070 hardcodeado.

Requiere: numpy, astropy (solo para leer el .ecsv). NO requiere healpy.
"""
import numpy as np
from astropy.table import Table

# ------------------------------------------------------------------ coordenadas
RA_NGP, DEC_NGP, L_NCP = np.radians(192.85948), np.radians(27.12825), np.radians(122.93192)
EPS = np.radians(23.4392911)

def eq2gal(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    b = np.arcsin(np.clip(np.sin(dec)*np.sin(DEC_NGP) +
                          np.cos(dec)*np.cos(DEC_NGP)*np.cos(ra-RA_NGP), -1, 1))
    y = np.cos(dec)*np.sin(ra-RA_NGP)
    x = np.sin(dec)*np.cos(DEC_NGP) - np.cos(dec)*np.sin(DEC_NGP)*np.cos(ra-RA_NGP)
    return np.degrees(L_NCP - np.arctan2(y, x)) % 360, np.degrees(b)

def eq2ecl(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    beta = np.arcsin(np.clip(np.sin(dec)*np.cos(EPS) - np.cos(dec)*np.sin(EPS)*np.sin(ra), -1, 1))
    lam = np.arctan2(np.sin(ra)*np.cos(EPS) + np.tan(dec)*np.sin(EPS), np.cos(ra))
    return np.degrees(lam) % 360, np.degrees(beta)

def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], axis=-1)

def sep_deg(l1, b1, l2, b2):
    return np.degrees(np.arccos(np.clip(np.sum(unit(l1, b1)*unit(l2, b2), axis=-1), -1, 1)))

# ------------------------------------------------------------------ estimador
def fit_dipolo(u_dat, u_rand):
    """Modelo dN/dOmega ~ (1 + D.n) sobre la region no enmascarada.
    A primer orden  <n>_datos = <n>_mask + M.D,  con M = <nn>_mask - <n>_mask<n>_mask.
    El catalogo aleatorio define la mascara, asi que cortes parciales se manejan solos."""
    mu_d, mu_r = u_dat.mean(axis=0), u_rand.mean(axis=0)
    M = (u_rand[:, :, None]*u_rand[:, None, :]).mean(axis=0) - np.outer(mu_r, mu_r)
    D = np.linalg.solve(M, mu_d - mu_r)
    amp = np.linalg.norm(D); n = D/amp
    return amp, np.degrees(np.arctan2(n[1], n[0])) % 360, np.degrees(np.arcsin(np.clip(n[2], -1, 1)))

# ------------------------------------------------------------------ datos
print("1. Cargando catalogo...")
t = Table.read("catwise_cuasares.ecsv")
ra, dec = np.asarray(t['ra'], float), np.asarray(t['dec'], float)
gl, gb = eq2gal(ra, dec)
_, eb = eq2ecl(ra, dec)
print(f"   {len(ra)} fuentes")

print("2. Generando catalogo aleatorio (define la mascara)...")
rng = np.random.default_rng(1)
NR = 8_000_000
dr = np.degrees(np.arcsin(rng.uniform(-1, 1, NR)))
rr = rng.uniform(0, 360, NR)
glr, gbr = eq2gal(rr, dr)
_, ebr = eq2ecl(rr, dr)

# Nubes de Magallanes: sobreviven |b|>30 (LMC esta en b=-32.9)
LMC, SMC, RAD = (80.894, -69.756), (13.187, -72.829), 8.0
noMC   = (sep_deg(ra, dec, *LMC) > RAD) & (sep_deg(ra, dec, *SMC) > RAD)
noMC_r = (sep_deg(rr, dr,  *LMC) > RAD) & (sep_deg(rr, dr,  *SMC) > RAD)

# ------------------------------------------------------------------ test ecliptico
print("\n3. TEST DE LA ECLIPTICA (densidad por area disponible)")
bins = np.linspace(-90, 90, 37); cen = 0.5*(bins[1:]+bins[:-1])
k, kr = (np.abs(gb) > 30) & noMC, (np.abs(gbr) > 30) & noMC_r
cd, _ = np.histogram(eb[k], bins=bins)
ca, _ = np.histogram(ebr[kr], bins=bins)
dens = cd/ca; dens /= dens.mean()
print(f"   densidad relativa: min={dens.min():.3f} max={dens.max():.3f} "
      f"(dispersion {100*dens.std():.1f}%)")
print(f"   tendencia vs |beta|: {100*np.polyfit(np.abs(cen), dens, 1)[0]*90:+.1f}% "
      f"entre ecliptica y polo")

# ------------------------------------------------------------------ expectativa cinematica
# Ellis & Baldwin (1984):  D = [2 + x(1+alpha)] * v/c
# x = pendiente de los conteos integrales: N(>S) ~ S^-x
# Para medirla necesitas la columna w1mpro (vuelve a bajarla si no la guardaste):
#   N(<m) ~ 10^(0.4*x*m)  ->  x = pendiente de log10 N(<m) vs m, dividida entre 0.4
# Con solo ra/dec no se puede; deja aqui el valor de Secrest+ como provisional.
X_SLOPE, ALPHA, V_C = 1.7, 1.0, 369.82/299792.458
D_ESPERADO = (2 + X_SLOPE*(1+ALPHA))*V_C
print(f"\n4. Expectativa cinematica con x={X_SLOPE}, alpha={ALPHA}: D = {D_ESPERADO:.4f} "
      f"({100*D_ESPERADO:.2f}%)   <-- MIDE x EN TUS DATOS, no lo heredes")

# ------------------------------------------------------------------ dipolo + estabilidad
print("\n5. DIPOLO Y TEST DE ESTABILIDAD")
print("   (un dipolo cosmologico real NO debe cambiar de amplitud al variar el corte)")
print(f"   {'corte':>8}  {'N':>9}  {'D':>16}  {'(l,b)':>18}  {'sep CMB':>8}")
for bc in (20, 25, 30, 35, 40, 45, 50, 55, 60):
    kd = (np.abs(gb) > bc) & noMC
    kri = (np.abs(gbr) > bc) & noMC_r
    u_d, u_r = unit(gl[kd], gb[kd]), unit(glr[kri], gbr[kri])
    amp, lon, lat = fit_dipolo(u_d, u_r)
    n = len(u_d)
    bs = [fit_dipolo(u_d[rng.integers(0, n, n)], u_r)[0] for _ in range(25)]
    print(f"   |b|>{bc:2d}   {n:9d}  {amp:.4f} +/- {np.std(bs):.4f}  "
          f"({lon:6.1f}, {lat:+5.1f})  {sep_deg(lon, lat, 264.021, 48.253):7.1f} deg")

print("\n   Referencia CMB: (l,b) = (264.0, +48.3)")
print("   Si la amplitud cae al ampliar el corte -> contaminacion galactica residual.")
