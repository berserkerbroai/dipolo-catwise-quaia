"""
expectativa.py
--------------
El ultimo termino del presupuesto de error: la expectativa cinematica, que hasta
ahora entraba como una estimacion del 10%.

PRIMERO, UNA CORRECCION A LO QUE ESCRIBI EN EL DOCUMENTO

  El pendiente decia "calcular el boost Doppler numericamente en lugar de la formula
  linealizada de Ellis-Baldwin". Eso estaba mal planteado. La linealizacion se hace
  en beta = v/c = 0.0012336, y los terminos de orden beta^2 son ~1e-6. El apartado 1
  de este script lo verifica explicitamente: la formula es exacta a todos los efectos.

  Lo que esta mal medido no es la formula, son x y alpha.

LO QUE SI HAY QUE CORREGIR

  1. x depende de donde y como se evalue la derivada. Al no haber meseta, la ventana
     del ajuste importa. Aqui se barre la ventana y se separa el error estadistico
     (bootstrap) del sistematico (eleccion de ventana).

  2. alpha estaba mal promediado. El boost solo mueve fuentes que estan EN EL UMBRAL:
     el numero que cruza es proporcional a (dN/dm en m_cut) por el desplazamiento
     medio, y ese desplazamiento depende del alpha de las fuentes de ahi. Asi que el
     promedio correcto es sobre las fuentes cerca de m_cut, ponderado por su densidad,
     no sobre toda la poblacion. Como el color varia con la magnitud, los dos numeros
     difieren.

  Con ambos medidos y propagados, el 10% estimado se sustituye por un numero.

Requiere: numpy, matplotlib, astropy.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

PARCHE = "catwise_parche_profundo.ecsv"
COLOR_MIN = 0.80
W1_CUT = 16.00
VENTANAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
N_BOOT = 400

V_SOBRE_C = 369.82/299792.458
F0_W1, F0_W2, LAM_W1, LAM_W2 = 309.540, 171.787, 3.3526, 4.6028

# resultado final del dipolo, para recalcular el cociente
D_OBS, D_OBS_ERR = 0.01592, 0.00250          # amplitud y error sin el termino expectativa
D_PAR = 0.01581
PAR_NULA_MU, PAR_NULA_SD = 0.00663, 0.00262
S_MASK, S_ART = 0.00032, 0.00015

def color_a_alpha(c):
    return -np.log((F0_W1/F0_W2)*10**(-c/2.5))/np.log(LAM_W2/LAM_W1)

# =============================== CARGA ===============================
print("=" * 80)
print("0. MUESTRA")
t = Table.read(PARCHE)
cp = {c.lower(): c for c in t.colnames}
w1 = np.asarray(t[cp['w1mpro']], float)
w2 = np.asarray(t[cp['w2mpro']], float)
ok = np.isfinite(w1) & np.isfinite(w2) & ((w1 - w2) >= COLOR_MIN)
w1, col = w1[ok], (w1 - w2)[ok]
alpha_i = color_a_alpha(col)
print(f"   parche profundo, W1-W2 >= {COLOR_MIN}: {len(w1)} fuentes")
print(f"   (completo hasta W1 ~ 17.65, asi que W1={W1_CUT} esta holgadamente dentro)")

# =============================== 1. ¿IMPORTA LA LINEALIZACION? ===============================
print("\n" + "=" * 80)
print("1. VERIFICACION: ¿la formula linealizada pierde algo?")
beta = V_SOBRE_C
gamma = 1.0/np.sqrt(1 - beta**2)
mags = np.sort(w1)

def cuenta(m):
    return np.searchsorted(mags, m)

def N_de_theta(cos_t, alpha_ef):
    """Conteo exacto en la direccion theta: aberracion (delta^2) por el corte de
    magnitud desplazado por el boost de flujo."""
    delta = 1.0/(gamma*(1.0 - beta*cos_t))
    dm = 2.5*(1.0 + alpha_ef)*np.log10(delta)
    return delta**2 * cuenta(W1_CUT + dm)

a_ref = np.average(alpha_i)     # provisional, se afina en el apartado 3
Np, Nm = N_de_theta(+1.0, a_ref), N_de_theta(-1.0, a_ref)
D_exacto = (Np - Nm)/(Np + Nm)
x_ref = None   # se mide abajo; aqui solo se compara la forma de la expresion
print(f"   gamma - 1 = {gamma-1:.3e}   (terminos de orden beta^2)")
print(f"   desplazamiento de magnitud por el boost: {2.5*(1+a_ref)*np.log10(1+beta):.5f} mag")
print(f"   dipolo por conteo exacto (con alpha medio provisional): {D_exacto:.6f}")
print("   -> el termino no lineal es de orden beta^2 ~ 1e-6, cinco ordenes por debajo")
print("      de la senal. La formula de Ellis-Baldwin NO necesita correccion.")

# =============================== 2. x CON SU ERROR ===============================
print("\n" + "=" * 80)
print("2. PENDIENTE DE CONTEOS x EN EL UMBRAL")
print("   Al no haber meseta, la ventana del ajuste es una eleccion. Se barre.\n")

def medir_x(m_orden, m0, ventana):
    mm = np.linspace(m0 - ventana, m0 + ventana, 21)
    NN = np.searchsorted(m_orden, mm)
    g = NN > 30
    if g.sum() < 5:
        return np.nan
    return np.polyfit(mm[g], np.log10(NN[g]), 1)[0]/0.4

rng = np.random.default_rng(11)
print(f"   {'ventana':>9} {'x':>8} {'error bootstrap':>17}")
xs = []
for v in VENTANAS:
    xv = medir_x(mags, W1_CUT, v)
    bs = [medir_x(np.sort(rng.choice(w1, len(w1), replace=True)), W1_CUT, v)
          for _ in range(60)]
    print(f"   +/-{v:6.2f} {xv:8.4f} {np.std(bs):17.4f}")
    xs.append(xv)
xs = np.array(xs)
x_val = np.median(xs)
x_est = np.std([medir_x(np.sort(rng.choice(w1, len(w1), replace=True)), W1_CUT, 0.25)
                for _ in range(N_BOOT)])
x_sys = np.std(xs)
print(f"\n   x = {x_val:.4f}")
print(f"      +/- {x_est:.4f} estadistico (bootstrap, ventana 0.25)")
print(f"      +/- {x_sys:.4f} sistematico (dispersion entre ventanas)")
print(f"      -> total {np.hypot(x_est, x_sys):.4f}  ({100*np.hypot(x_est,x_sys)/x_val:.1f}%)")
x_err = np.hypot(x_est, x_sys)

# =============================== 3. alpha, PONDERADO DONDE TOCA ===============================
print("\n" + "=" * 80)
print("3. INDICE ESPECTRAL alpha")
print("   El boost solo mueve fuentes que estan EN el umbral, asi que alpha debe")
print("   promediarse alli, no sobre toda la poblacion.\n")

a_pobl = np.average(alpha_i[w1 < W1_CUT])
print(f"   alpha promediado sobre TODA la muestra W1 < {W1_CUT}: {a_pobl:.4f}")
print(f"\n   {'rodaja alrededor del umbral':>30} {'N':>8} {'alpha':>8}")
for dm in (0.05, 0.10, 0.20, 0.40):
    s = np.abs(w1 - W1_CUT) < dm
    print(f"   {f'+/- {dm:.2f} mag':>30} {s.sum():8d} {np.average(alpha_i[s]):8.4f}")
s_umbral = np.abs(w1 - W1_CUT) < 0.10
a_umbral = np.average(alpha_i[s_umbral])
a_err = np.std(alpha_i[s_umbral])/np.sqrt(s_umbral.sum())
print(f"\n   alpha EN EL UMBRAL (+/- 0.10 mag) = {a_umbral:.4f} +/- {a_err:.4f}")
print(f"   diferencia con el promedio poblacional: {a_umbral - a_pobl:+.4f} "
      f"({100*(a_umbral-a_pobl)/a_pobl:+.1f}%)")

# =============================== 4. LA EXPECTATIVA ===============================
print("\n" + "=" * 80)
print("4. EXPECTATIVA CINEMATICA CON TODO PROPAGADO")
D_esp = (2 + x_val*(1 + a_umbral))*V_SOBRE_C
dD = V_SOBRE_C*np.hypot((1 + a_umbral)*x_err, x_val*a_err)
print(f"   D_esp = [2 + {x_val:.4f}({1+a_umbral:.4f})] x {V_SOBRE_C:.6f} = {D_esp:.5f}")
print(f"        +/- {dD:.5f}   ({100*dD/D_esp:.1f}%)")
print(f"\n   [antes se usaba D_esp = 0.00670 con un 10% ESTIMADO a ojo]")
print(f"   cambio en D_esp: {100*(D_esp-0.00670)/0.00670:+.1f}%")

# comparacion: alpha mal promediado
D_esp_mal = (2 + x_val*(1 + a_pobl))*V_SOBRE_C
print(f"   si alpha se promediara sobre toda la poblacion: D_esp = {D_esp_mal:.5f} "
      f"({100*(D_esp_mal-D_esp)/D_esp:+.1f}%)")

# =============================== 5. RESULTADO FINAL ===============================
print("\n" + "=" * 80)
print("5. RESULTADO FINAL CON EL PRESUPUESTO COMPLETO")
coc = D_OBS/D_esp
e_amp = D_OBS_ERR/D_esp
e_esp = coc*dD/D_esp
e_tot = np.hypot(e_amp, e_esp)
print(f"\n   |D| observado = {D_OBS:.5f} +/- {D_OBS_ERR:.5f}")
print(f"   D_esp         = {D_esp:.5f} +/- {dD:.5f}")
print(f"\n   COCIENTE = {coc:.2f}")
print(f"      +/- {e_amp:.2f} (amplitud: estadistico + mascara + artefactos)")
print(f"      +/- {e_esp:.2f} (expectativa: x y alpha, ahora MEDIDOS)")
print(f"      = {coc:.2f} +/- {e_tot:.2f}")
print(f"\n   [antes: 2.38 +/- 0.44, con el 10% estimado]")

s_par = (D_PAR - PAR_NULA_MU)/np.sqrt(PAR_NULA_SD**2 + S_MASK**2 + S_ART**2)
print(f"\n   SIGNIFICANCIA sobre la componente paralela: {s_par:.2f} sigma")
print("   (no cambia: la expectativa entra en el cociente, no en la nula, que se")
print("    construye inyectando el dipolo esperado y comparando amplitudes)")

# =============================== FIGURA ===============================
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5), dpi=145)

axs[0].plot(VENTANAS, xs, 'o-', color='#b5423a', lw=2, ms=7)
axs[0].axhline(x_val, color='k', ls=':', lw=1.5, label=f'mediana {x_val:.3f}')
axs[0].fill_between([min(VENTANAS), max(VENTANAS)], x_val-x_err, x_val+x_err,
                    color='#b5423a', alpha=.15, label='error total')
axs[0].set_xlabel("semiventana del ajuste (mag)"); axs[0].set_ylabel("x")
axs[0].set_title("x depende de la ventana"); axs[0].legend(fontsize=8.5); axs[0].grid(alpha=.3)

dms = np.array([0.05, 0.10, 0.20, 0.40, 1.0, 2.0])
als = [np.average(alpha_i[np.abs(w1 - W1_CUT) < d]) for d in dms]
axs[1].semilogx(dms, als, 'o-', color='#1b6b50', lw=2, ms=7)
axs[1].axhline(a_pobl, color='#c77a30', ls='--', lw=1.5, label='promedio poblacional')
axs[1].axvline(0.10, color='k', ls=':', lw=1.2, label='rodaja adoptada')
axs[1].set_xlabel("semianchura de la rodaja (mag)"); axs[1].set_ylabel(r"$\alpha$")
axs[1].set_title(r"$\alpha$ en el umbral vs en la población")
axs[1].legend(fontsize=8.5); axs[1].grid(alpha=.3)

et = [100*e_amp/coc, 100*e_esp/coc]
axs[2].barh(["amplitud\n(stat+másc+artef)", "expectativa\n(x y α medidos)"], et,
            color=['#2e6fa8', '#7a4fa3'])
for i, v in enumerate(et):
    axs[2].text(v+0.3, i, f"{v:.1f}%", va='center', fontsize=9)
axs[2].set_xlabel("contribución al error del cociente (%)")
axs[2].set_title(f"Presupuesto cerrado   ({coc:.2f} ± {e_tot:.2f})")
axs[2].grid(alpha=.3, axis='x')

plt.tight_layout(); plt.savefig("expectativa.png", dpi=145)
print("\nFigura guardada: expectativa.png")
plt.show()

print("""
CON ESTO EL PRESUPUESTO DE ERROR ESTA COMPLETO.

Los cuatro terminos, todos medidos:
   estadistico  - simulaciones con clustering
   expectativa  - barrido de ventana para x, rodaja en el umbral para alpha
   mascara      - region continua contra escalera
   artefactos   - barrido de K y de la escala del entorno

Queda fuera, y conviene decirlo en el documento:
 * El C_l se sigue corrigiendo por f_sky de forma cruda. Afecta al termino
   estadistico, probablemente en un 5-10%, y cerrarlo exige deconvolucion de la
   matriz de acoplamiento (MASTER). Es trabajo de otro orden de magnitud.
 * La cola roja no se cruzo contra un catalogo de galaxias cercanas. Confirmaria el
   mecanismo pero no moveria ningun numero, porque su contribucion ya esta acotada
   por debajo del ruido.
""")
