import os
import numpy as np
import matplotlib.pyplot as plt
import pyvo as vo
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

ARCHIVO_MAG = "catwise_con_magnitud.ecsv"
service = vo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")

def bajar_cuadrante_mag(nombre, condicion):
    query = f"""
    SELECT ra, dec, w1mpro 
    FROM catwise_2020 
    WHERE w1mpro - w2mpro >= 0.8 
    AND w1snr >= 5 
    AND w2snr >= 5
    AND w1mpro < 16.5
    AND {condicion}
    """
    print(f"[{nombre}] Escaneando magnitudes en la NASA...")
    job = service.submit_job(query)
    job.execution_duration = 3600
    job.run()
    job.wait(phases=['COMPLETED', 'ERROR', 'ABORTED'])
    if job.phase == 'COMPLETED':
        return job.fetch_result().to_table()
    else:
        job.raise_if_error()
        return None

try:
    if os.path.exists(ARCHIVO_MAG):
        print(f"1. Cargando {ARCHIVO_MAG} desde disco duro...")
        t = Table.read(ARCHIVO_MAG)
    else:
        print("1. Descargando catálogo con columna w1mpro (tomará unos minutos)...")
        cuadrantes = [
            ("Norte 1", "dec >= 0 AND ra < 180"), ("Norte 2", "dec >= 0 AND ra >= 180"),
            ("Sur 1",   "dec < 0 AND ra < 180"),   ("Sur 2",   "dec < 0 AND ra >= 180")
        ]
        tablas = [bajar_cuadrante_mag(n, c) for n, c in cuadrantes]
        t = vstack(tablas)
        t.write(ARCHIVO_MAG, overwrite=True)
        print("   Catálogo con magnitudes guardado localmente.")

    print("\n2. Filtrando galaxia (|b| > 30) para medir solo cuásares puros...")
    ra, dec, w1 = t['ra'].data, t['dec'].data, t['w1mpro'].data
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    mask_b = np.abs(coords.galactic.b.degree) > 30.0
    w1_limpio = w1[mask_b]

    print("3. Calculando la pendiente de los conteos integrales (x)...")
    # Ordenamos las magnitudes de la más brillante a la más débil
    mags = np.sort(w1_limpio)
    # Contamos cuántos objetos hay más brillantes que cada magnitud (N < m)
    N_menor_que_m = np.arange(1, len(mags) + 1)
    logN = np.log10(N_menor_que_m)

    # El rango lineal seguro suele estar entre la magnitud 15.0 y justo antes de nuestro corte (16.4)
    mascara_ajuste = (mags >= 15.0) & (mags <= 16.4)
    mags_fit = mags[mascara_ajuste]
    logN_fit = logN[mascara_ajuste]

    # Ajuste lineal: log10(N(<m)) = pendiente * m + intercepto
    pendiente, intercepto = np.polyfit(mags_fit, logN_fit, 1)
    
    # En astrofísica, la pendiente teórica x es la pendiente ajustada dividida entre 0.4
    x_medido = pendiente / 0.4

    # Calculamos la expectativa cinemática real usando TUS datos
    ALPHA = 1.0 # Índice espectral promedio para cuásares
    V_C = 369.82 / 299792.458 # Velocidad del sistema solar respecto al CMB / velocidad de la luz
    D_esperado = (2 + x_medido * (1 + ALPHA)) * V_C

    print("-" * 50)
    print(f"PARÁMETROS CINEMÁTICOS CALCULADOS DESDE TUS DATOS:")
    print(f"Pendiente de conteos (x) : {x_medido:.4f}")
    print(f"Expectativa Teórica (D)  : {D_esperado:.4f} ({(D_esperado*100):.2f}%)")
    print("-" * 50)

    # Visualizamos el ajuste para comprobar que sea una línea recta perfecta en ese rango
    plt.figure(figsize=(8, 5))
    plt.plot(mags, logN, color='gray', label='Datos Crudos (log N < m)', alpha=0.5)
    plt.plot(mags_fit, logN_fit, color='blue', linewidth=3, label='Rango de Ajuste Lineal (15.0 - 16.4)')
    plt.plot(mags_fit, pendiente * mags_fit + intercepto, color='red', linestyle='--', label=f'Ajuste (x={x_medido:.2f})')
    plt.axvline(16.5, color='black', linestyle=':', label='Corte de Catálogo (16.5)')
    plt.title(f"Distribución de Magnitudes Infrarrojas (CatWISE w1mpro)")
    plt.xlabel("Magnitud W1")
    plt.ylabel("log10( N < W1 )")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

except Exception as e:
    print(f"Error: {e}")