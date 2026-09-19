import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

ARCHIVO_LOCAL = "catwise_cuasares.ecsv"

print("1. Cargando datos locales desde caché...")
tabla_maestra = Table.read(ARCHIVO_LOCAL)
ra_real = tabla_maestra['ra'].data
dec_real = tabla_maestra['dec'].data

print("2. Convirtiendo a coordenadas Galácticas y Eclípticas (para diagnóstico)...")
coords = SkyCoord(ra=ra_real*u.degree, dec=dec_real*u.degree, frame='icrs')
gal_coords = coords.galactic
ecl_coords = coords.barycentrictrueecliptic

# === EL TEST DE LA ECLÍPTICA (Diagnóstico de sistemáticas de WISE) ===
print("3. Generando diagnóstico de escaneo del satélite WISE...")
# Quitamos la franja galáctica solo para ver el cielo profundo limpio
mask_gal_bruta = np.abs(gal_coords.b.degree) > 30.0
lat_ecl_limpias = ecl_coords.lat.degree[mask_gal_bruta]

plt.figure(figsize=(10, 5))
plt.hist(lat_ecl_limpias, bins=50, color='purple',
         alpha=0.7, edgecolor='black')
plt.title("Densidad de Cuásares vs Latitud Eclíptica (Buscando la huella de WISE)")
plt.xlabel("Latitud Eclíptica (grados)")
plt.ylabel("Número de Cuásares observados")
plt.grid(True, alpha=0.3)
plt.show()  # <--- CIERRA ESTA VENTANA PARA CONTINUAR AL CÁLCULO DEL DIPOLO

# === CORRECCIÓN DE MÁSCARA Y COORDENADAS ===
print("4. Construyendo mapa y máscara HEALPix estricta (Bug de borde resuelto)...")
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)

# Creamos un mapa vacío
density_map = np.zeros(NPIX)

# Convertimos a radianes DIRECTAMENTE EN GALÁCTICAS para el mapa nativo
theta_gal = np.radians(90.0 - gal_coords.b.degree)
phi_gal = np.radians(gal_coords.l.degree)

# Llenamos los píxeles sumando 1 por cada cuásar
pixel_indices = hp.ang2pix(NSIDE, theta_gal, phi_gal)
np.add.at(density_map, pixel_indices, 1)

# Construimos la máscara usando la geometría de los CENTROS de los píxeles
theta_pix, phi_pix = hp.pix2ang(NSIDE, np.arange(NPIX))
lat_pix = 90.0 - np.degrees(theta_pix)

# Máscara estricta: descartamos todo píxel cuyo centro esté a menos de 30° del ecuador
mask_array = np.abs(lat_pix) <= 30.0
density_map_masked = hp.ma(density_map)
density_map_masked.mask = mask_array

print("\n5. Recalculando el Dipolo en el marco Galáctico puro...")
monopole, dipole_vector = hp.fit_dipole(density_map_masked)
amplitud_dipolo = np.linalg.norm(dipole_vector) / monopole

vec_norm = dipole_vector / np.linalg.norm(dipole_vector)
theta_dip, phi_dip = hp.vec2dir(vec_norm)
lon_gal_medida = np.degrees(phi_dip)
lat_gal_medida = 90.0 - np.degrees(theta_dip)

print("-" * 50)
print("RESULTADOS DEL DIPOLO CORREGIDO (CatWISE):")
print(
    f"Amplitud real              : {amplitud_dipolo:.4f} ({(amplitud_dipolo*100):.2f}%)")
print(
    f"Dirección Galáctica (l, b) : ({lon_gal_medida:.2f}°, {lat_gal_medida:.2f}°)")
print("-" * 50)
print("Comparación contra el CMB (Cinemática Estándar):")
print("Dirección esperada (l, b)  : (264.0°, 48.2°)")

# Distancia angular matemática exacta usando Astropy
coord_medida = SkyCoord(l=lon_gal_medida*u.degree,
                        b=lat_gal_medida*u.degree, frame='galactic')
coord_cmb = SkyCoord(l=264.0*u.degree, b=48.2*u.degree, frame='galactic')
separacion = coord_medida.separation(coord_cmb).degree

print(f"Separación angular real    : {separacion:.2f} grados de diferencia")
print("-" * 50)

print("\n6. Generando mapa final corregido...")
hp.mollview(density_map_masked,
            title="CatWISE 2020: Mapa Galáctico Corregido", coord=['G'], cmap='magma')
plt.show()
