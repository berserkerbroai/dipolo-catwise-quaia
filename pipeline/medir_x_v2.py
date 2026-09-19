"""
medir_x_v2.py
-------------
Mide los DOS parametros que entran en la expectativa cinematica de Ellis & Baldwin:

    D_esperado = [ 2 + x (1 + alpha) ] * v/c

    x     = pendiente LOCAL de los conteos integrales EN EL LIMITE DE FLUJO
            (definida por N(>S) ~ S^-x ;  con S ~ 10^-0.4m  =>  x = pendiente(log10 N(<m) vs m)/0.4)
    alpha = indice espectral en la banda de observacion (S_nu ~ nu^-alpha),
            medido del color W1-W2 de la propia muestra.

Cambios respecto a medir_x.py:
  1. PARCHE PROFUNDO en vez de cielo completo al mismo corte.
     La pendiente de conteos es propiedad de la POBLACION, no de la cobertura angular.
     Bajar ~1000 deg^2 hasta W1<18 es mas rapido que 40000 deg^2 hasta 16.5, y ademas
     permite VER donde empieza la incompletitud. Con el corte en 16.5 eso es imposible:
     los datos se acaban justo donde necesitas mirar.
  2. PENDIENTE LOCAL, no promediada. Ellis-Baldwin necesita la derivada en el umbral,
     porque el boost Doppler actua empujando fuentes a traves de ese umbral. Promediar
     1.4 magnitudes de una curva que se aplana da un numero distinto.
  3. DIAGNOSTICO DE COMPLETITUD explicito: x_local(m) debe tener una meseta. Si cae
     antes de tu corte, tu muestra del dipolo esta incompleta y hay que subir el corte.
  4. alpha MEDIDO del color, no asumido = 1.0.
  5. Errores por bootstrap y propagacion a D_esperado.

Requiere: numpy, matplotlib, astropy, pyvo.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# =============================== CONFIGURACION ===============================
ARCHIVO = "catwise_parche_profundo.ecsv"

# Parche a |b| galactica alta (b entre +50 y +78), ~1034 deg^2 (2.5% del cielo).
RA_MIN, RA_MAX   = 150.0, 210.0
DEC_MIN, DEC_MAX =  20.0,  40.0

W1_MAX_DESCARGA = 18.0   # bajamos MUCHO mas profundo que el corte de ciencia
W1_CORTE_CIENCIA = 16.5  # el corte que usas en el analisis del dipolo

# Zero points WISE (Vega), Explanatory Supplement
F0_W1, F0_W2 = 309.540, 171.787          # Jy
LAM_W1, LAM_W2 = 3.3526, 4.6028          # micras
V_SOBRE_C = 369.82 / 299792.458          # dipolo CMB

# Tu medicion limpia del dipolo (dipolo_v3, |b|>30, rechazo 12sigma)
D_OBSERVADO, D_OBSERVADO_ERR = 0.0169, 0.0013

# =============================== DESCARGA ===============================
def descargar():
    import pyvo as vo
    from astropy.table import Table
    service = vo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
    query = f"""
    SELECT ra, dec, w1mpro, w2mpro
    FROM catwise_2020
    WHERE w1mpro - w2mpro >= 0.8
      AND w1snr >= 5 AND w2snr >= 5
      AND w1mpro < {W1_MAX_DESCARGA}
      AND ra  BETWEEN {RA_MIN} AND {RA_MAX}
      AND dec BETWEEN {DEC_MIN} AND {DEC_MAX}
    """
    print(f"Descargando parche de ~1034 deg^2 hasta W1<{W1_MAX_DESCARGA}...")
    job = service.submit_job(query)
    job.execution_duration = 3600
    job.run()
    job.wait(phases=['COMPLETED', 'ERROR', 'ABORTED'])
    if job.phase != 'COMPLETED':
        job.raise_if_error()
        raise RuntimeError(f"Fase final: {job.phase}")
    t = job.fetch_result().to_table()
    t.write(ARCHIVO, overwrite=True)
    print(f"   Guardado en {ARCHIVO}")
    return t

from astropy.table import Table
if os.path.exists(ARCHIVO):
    print(f"1. Cargando {ARCHIVO} desde disco...")
    t = Table.read(ARCHIVO)
else:
    print("1. No hay cache local.")
    t = descargar()

w1 = np.asarray(t['w1mpro'], float)
w2 = np.asarray(t['w2mpro'], float)
ok = np.isfinite(w1) & np.isfinite(w2)
w1, w2 = w1[ok], w2[ok]
print(f"   {len(w1)} fuentes en el parche\n")

# =============================== 2. PENDIENTE LOCAL x(m) ===============================
print("2. Midiendo la pendiente LOCAL de los conteos, x(m)...")

def slope_local(mags, m_grid, media_ventana=0.25):
    """Pendiente local de log10 N(<m) vs m, por minimos cuadrados en una ventana
    deslizante de +/- media_ventana magnitudes. Devuelve x = pendiente/0.4."""
    mags = np.sort(mags)
    out = np.full(len(m_grid), np.nan)
    for i, m0 in enumerate(m_grid):
        lo, hi = m0 - media_ventana, m0 + media_ventana
        # N(<m) evaluado en una rejilla fina dentro de la ventana
        mm = np.linspace(lo, hi, 15)
        NN = np.searchsorted(mags, mm)            # cuenta acumulada
        good = NN > 30                            # evita log de numeros diminutos
        if good.sum() < 5:
            continue
        p = np.polyfit(mm[good], np.log10(NN[good]), 1)
        out[i] = p[0] / 0.4
    return out

m_grid = np.arange(13.0, W1_MAX_DESCARGA - 0.3, 0.05)
x_loc = slope_local(w1, m_grid)

# bootstrap
rng = np.random.default_rng(0)
NB = 30
boot = np.empty((NB, len(m_grid)))
for b in range(NB):
    boot[b] = slope_local(w1[rng.integers(0, len(w1), len(w1))], m_grid)
x_err = np.nanstd(boot, axis=0)

# diagnostico de completitud: donde x_loc alcanza su maximo y donde cae un 10%
ival = np.isfinite(x_loc)
i_max = np.nanargmax(np.where(m_grid > 14.0, x_loc, np.nan))
m_pico, x_pico = m_grid[i_max], x_loc[i_max]
cae = np.where((m_grid > m_pico) & (x_loc < 0.90 * x_pico))[0]
m_rotura = m_grid[cae[0]] if len(cae) else np.nan

print(f"   x_local alcanza su maximo en W1 = {m_pico:.2f}  (x = {x_pico:.3f})")
if np.isfinite(m_rotura):
    print(f"   x_local cae por debajo del 90% del pico en W1 = {m_rotura:.2f}")
    print(f"   -> la muestra deja de ser completa alrededor de W1 ~ {m_rotura:.2f}")
else:
    print("   x_local no cae: la muestra parece completa en todo el rango medido")

# valor de x en el corte de ciencia
j = np.argmin(np.abs(m_grid - W1_CORTE_CIENCIA))
x_corte, x_corte_err = x_loc[j], x_err[j]
print(f"\n   x en tu corte de ciencia (W1={W1_CORTE_CIENCIA}): "
      f"{x_corte:.3f} +/- {x_corte_err:.3f}")

if np.isfinite(m_rotura) and m_rotura < W1_CORTE_CIENCIA:
    print(f"   *** AVISO: la incompletitud empieza ANTES de {W1_CORTE_CIENCIA}.")
    print(f"       El x medido ahi esta sesgado hacia abajo. Considera recortar la")
    print(f"       muestra del dipolo en W1 < {m_rotura:.1f} y volver a medir todo.")

# =============================== 3. alpha DEL COLOR ===============================
print("\n3. Midiendo alpha del color W1-W2...")
sel = w1 < W1_CORTE_CIENCIA          # mismo corte que la muestra del dipolo
color = (w1 - w2)[sel]

def color_a_alpha(c):
    """S_nu ~ nu^-alpha.  S_W1/S_W2 = (F0_W1/F0_W2)*10^(-(W1-W2)/2.5)
       y  S_W1/S_W2 = (nu_W1/nu_W2)^-alpha,  con nu_W1/nu_W2 = lam_W2/lam_W1."""
    ratio = (F0_W1 / F0_W2) * 10 ** (-c / 2.5)
    return -np.log(ratio) / np.log(LAM_W2 / LAM_W1)

alpha_med = color_a_alpha(np.median(color))
alpha_sig = 0.5 * (color_a_alpha(np.percentile(color, 84)) -
                   color_a_alpha(np.percentile(color, 16)))
alpha_err_media = alpha_sig / np.sqrt(len(color))   # error de la mediana

print(f"   color W1-W2: mediana = {np.median(color):.3f}, "
      f"dispersion 16-84% = {np.percentile(color,84)-np.percentile(color,16):.3f}")
print(f"   alpha = {alpha_med:.3f}  (dispersion de la poblacion {alpha_sig:.3f}, "
      f"error de la mediana {alpha_err_media:.4f})")

# =============================== 4. EXPECTATIVA ===============================
print("\n4. Expectativa cinematica de Ellis-Baldwin con TUS parametros")
D_esp = (2 + x_corte * (1 + alpha_med)) * V_SOBRE_C
# propagacion: el termino dominante suele ser la dispersion de alpha, no su error estandar,
# porque la muestra tiene un rango real de indices espectrales. Damos las dos versiones.
dD_estad = V_SOBRE_C * np.hypot((1 + alpha_med) * x_corte_err, x_corte * alpha_err_media)
dD_pobl  = V_SOBRE_C * np.hypot((1 + alpha_med) * x_corte_err, x_corte * alpha_sig)

print(f"   x     = {x_corte:.3f} +/- {x_corte_err:.3f}")
print(f"   alpha = {alpha_med:.3f}")
print(f"   D_esperado = {D_esp:.5f} ({100*D_esp:.3f}%)")
print(f"      +/- {dD_estad:.5f} (solo error estadistico)")
print(f"      +/- {dD_pobl:.5f} (si propagas la dispersion de alpha de la poblacion)")

exceso = D_OBSERVADO / D_esp
exceso_err = exceso * np.hypot(D_OBSERVADO_ERR / D_OBSERVADO, dD_estad / D_esp)
print(f"\n   D_observado = {D_OBSERVADO:.4f} +/- {D_OBSERVADO_ERR:.4f}")
print(f"   EXCESO = {exceso:.2f} +/- {exceso_err:.2f} veces la expectativa cinematica")
print(f"   (Secrest+2021 reportan ~2x con significancia 4.9 sigma)")

# =============================== 5. FIGURAS ===============================
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), dpi=140)

mags_s = np.sort(w1)
N = np.arange(1, len(mags_s) + 1)
axs[0].plot(mags_s[::200], np.log10(N[::200]), color='0.35', lw=2)
axs[0].axvline(W1_CORTE_CIENCIA, color='crimson', ls='--', lw=1.5,
               label=f'corte de ciencia ({W1_CORTE_CIENCIA})')
axs[0].set_xlabel("W1"); axs[0].set_ylabel(r"$\log_{10} N(<W1)$")
axs[0].set_title("Conteos integrales"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)

axs[1].fill_between(m_grid, x_loc - x_err, x_loc + x_err, color='steelblue', alpha=.3)
axs[1].plot(m_grid, x_loc, color='steelblue', lw=2)
axs[1].axvline(W1_CORTE_CIENCIA, color='crimson', ls='--', lw=1.5, label='corte de ciencia')
if np.isfinite(m_rotura):
    axs[1].axvline(m_rotura, color='darkorange', ls=':', lw=1.8,
                   label=f'incompletitud (~{m_rotura:.1f})')
axs[1].errorbar([W1_CORTE_CIENCIA], [x_corte], yerr=[x_corte_err], fmt='o',
                color='crimson', ms=7, zorder=5)
axs[1].set_xlabel("W1"); axs[1].set_ylabel("x local")
axs[1].set_title("Pendiente LOCAL: busca la meseta"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)

axs[2].hist(color, bins=80, color='mediumpurple', edgecolor='none')
axs[2].axvline(np.median(color), color='k', lw=1.8,
               label=f'mediana {np.median(color):.2f} -> alpha={alpha_med:.2f}')
axs[2].set_xlabel("W1 - W2"); axs[2].set_ylabel("N")
axs[2].set_title("Color -> indice espectral"); axs[2].legend(fontsize=8); axs[2].grid(alpha=.3)

plt.tight_layout()
plt.savefig("medir_x_diagnostico.png", dpi=140)
print("\nFigura guardada: medir_x_diagnostico.png")
plt.show()
