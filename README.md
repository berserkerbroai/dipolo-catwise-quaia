# Dipolo cósmico en CatWISE2020 y Quaia

Medición independiente del dipolo en la distribución angular de cuásares, con
presupuesto de error completo. **Es un estudio de replicación**: los resultados
reproducen mediciones ya publicadas y no constituyen evidencia nueva para el campo.
El valor de este repositorio es el método y su reproducibilidad.

---

## Resultado

Muestra CatWISE2020 con W1−W2 ≥ 0.80, W1 < 16.00, |b| > 30°, sin Nubes de
Magallanes, con rechazo de celdas atípicas por referencia local. N = 763 543.

| Cantidad | Valor |
| --- | --- |
| Amplitud del dipolo | 0.01592 ± 0.00250 |
| Dirección (l, b) | (258.2, +42.8) |
| Separación del dipolo CMB | 6.8° |
| Cociente sobre la expectativa cinemática | 2.38 ± 0.37 |
| Significancia frente a la nula cinemática | 3.47σ (componente paralela) |

Presupuesto de error sobre la amplitud, los cuatro términos medidos:

| Término | % | Método |
| --- | --- | --- |
| Estadístico | 15.5% | 2000 simulaciones con clustering + Poisson |
| Máscara | 2.0% | Región continua frente a escalera de píxeles |
| Expectativa cinemática | 1.0% | x con barrido de ventana, α en el umbral |
| Rechazo de artefactos | 0.9% | Barrido de K y de la escala del entorno |

En Quaia, con |b| > 40° y la función de selección publicada: 0.01121 ± 0.00308 a
2.4° del dipolo CMB (G20.0) y 0.01270 ± 0.00199 (G20.5). **No se detecta evolución
del dipolo con el corrimiento al rojo**; los dos catálogos dan diferencias entre
capas de z de signo opuesto, ninguna significativa.

Comparación con la literatura: Secrest et al. 2021 obtienen 0.0155 a ~27° con 4.9σ;
la reevaluación de 2025 sitúa la significancia en 3.27–3.63σ. Este trabajo cae en el
segundo rango.

---

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
cd pipeline
python estabilidad_corte.py    # descarga el cielo completo desde IRSA TAP (lento)
python medir_x_v2.py           # descarga el parche profundo
python contaminante.py
python cobertura.py            # descarga la muestra de cobertura
python test_final.py
python mascara_final.py
python barrido_artefactos.py
python barrido_color.py
python expectativa.py
python quaia_v2.py             # necesita los FITS de Quaia, ver abajo
```

`estabilidad_corte.py` es el punto de entrada: descarga el cielo completo con
magnitudes y genera `cielo_cache.npz`, del que dependen casi todos los demas.
Los scripts cachean sus descargas, así que solo la primera ejecución contacta con
los servidores. Esperan encontrar los archivos en el directorio desde el que se
ejecutan. Las descargas pesadas tardan decenas de minutos en la cola de IRSA.

**Los datos no están en el repositorio.** Son varios GB y se descargan solos, salvo
los de Quaia:

| Archivo | Origen | Tamaño |
| --- | --- | --- |
| `cielo_cache.npz` | IRSA TAP (automático) | 8 868 819 fuentes |
| `catwise_parche_profundo.ecsv` | IRSA TAP (automático) | 1 687 922 fuentes |
| `cobertura_cache.npz` | IRSA TAP (automático) | 899 535 fuentes |
| `quaia_G20.0.fits` | [Zenodo 10.5281/zenodo.10403370](https://doi.org/10.5281/zenodo.10403370) | 755 850 fuentes |
| `quaia_G20.5.fits` | ídem | 1 295 502 fuentes |
| `selection_function_NSIDE64_G20.0.fits` | ídem | HEALPix NSIDE 64 |

---

## Estructura

```
pipeline/       scripts que producen el resultado final
diagnostico/    scripts que encontraron y aislaron errores
figuras/        salidas gráficas
documento/      el informe completo con método, resultados y limitaciones
```

### pipeline/

| Script | Qué hace | Descarga |
| --- | --- | --- |
| `estabilidad_corte.py` | Punto de entrada. Descarga el cielo, D_obs y D_esp por corte de magnitud | `cielo_cache.npz` |
| `medir_x_v2.py` | Parche profundo; pendiente de conteos e índice espectral | `catwise_parche_profundo.ecsv` |
| `contaminante.py` | Bootstrap conjunto, capas disjuntas, dirección del contaminante | — |
| `cobertura.py` | Mecanismo del gradiente eclíptico; test diferencial de cobertura | `cobertura_cache.npz` |
| `test_final.py` | Rejilla de 32 configuraciones de corte | — |
| `mascara_final.py` | Test de sesgo de máscaras, significancia, presupuesto de error | — |
| `barrido_artefactos.py` | Barrido del umbral de rechazo de artefactos | — |
| `barrido_color.py` | Barrido del umbral superior de color | — |
| `expectativa.py` | Verifica la linealización; mide x y α; cierra el presupuesto | — |
| `quaia_v2.py` | Quaia con función de selección, capas de redshift | — |

### diagnostico/

Contienen errores que se encontraron y corrigieron, o quedaron superados por
versiones posteriores. Se conservan porque el informe los referencia y porque el
proceso de aislarlos es parte del método.

| Script | Qué pasó con él |
| --- | --- |
| `dipolo.py` | Primera versión. Pixelizaba en ecuatoriales y comparaba contra una dirección galáctica |
| `dipolo_v2.py` | Corrige el marco de coordenadas y la máscara de borde |
| `medir_x.py` | Primera versión. Medía la pendiente promediada, sobre un catálogo truncado en el propio corte |
| `quaia_dipolo_z.py` | Primera versión de Quaia. Máscara demasiado permisiva, bordes de z por cuantiles |
| `clustering.py` | Errores con clustering. Tenía el anillo de déficit en el borde de la máscara |
| `clustering_v2.py` | Corrige el borde; el umbral global de rechazo eliminaba el 5.5% de la máscara |
| `clustering_v3.py` | Referencia local para el rechazo. Superado por `mascara_final.py` |
| `reconciliar.py` | Aisló la discrepancia del 33% entre estimadores variando una cosa a la vez |

---

## Los tres errores que más costaron

Todos fueron geométricos, ninguno físico. Están documentados en detalle en el
informe; se resumen aquí porque es lo más transferible del trabajo.

1. **Marcos de coordenadas mezclados.** Pixelizar en ecuatoriales y comparar el
   resultado contra una dirección expresada en galácticas. Producía una separación
   aparente de 131.7° cuando la real era 43.2°.
2. **Cortes a nivel de fuente mezclados con cortes a nivel de píxel.** Filtrar las
   fuentes por su propia latitud galáctica mientras la máscara se evaluaba en los
   centros de píxel deja un anillo de déficit artificial en el borde. Infló el
   cuadrupolo del mapa por un factor 9 y la amplitud del dipolo un 33%.
3. **Catálogo aleatorio demasiado pequeño para representar la máscara.** Seis
   millones de puntos inyectan ±0.0011 en la amplitud, el 7% de la señal, sin
   aparecer en ninguna barra de error. Se resuelve con una rejilla determinista.

Y la regla que los resolvió: cuando dos métodos que deberían medir lo mismo
difieren, hay que variar una cosa a la vez hasta aislar la causa, no redefinir el
test para que coincidan. Un test que no puede fallar no valida nada.

---

## Limitaciones conocidas

- El C_ℓ se corrige por f_sky de forma cruda. Un tratamiento riguroso de cielo
  cortado con deconvolución de la matriz de acoplamiento daría un C_ℓ distinto a
  multipolos bajos, que es donde más pesa. Afecta al término estadístico,
  probablemente en un 5–10% de su valor.
- La pendiente de conteos x no presenta meseta: los conteos no siguen una ley de
  potencias en el rango relevante, que es lo que asume la fórmula de Ellis-Baldwin.
  El efecto sobre ventanas razonables es del 1.4% y está dentro del presupuesto.
- La cola roja del color está contaminada por fragmentación de galaxias cercanas.
  Su contribución está acotada en 0.00077, por debajo del suelo de ruido, y no se
  cruzó contra un catálogo de galaxias para confirmarlo objeto a objeto.

El informe completo en `documento/` detalla estas y otras.

---

## Datos y atribución

Este trabajo utiliza el catálogo CatWISE2020 (Marocco et al. 2021; Eisenhardt et al.
2020), descargado mediante el servicio TAP del NASA/IPAC Infrared Science Archive
(IRSA), operado por el Jet Propulsion Laboratory, California Institute of Technology,
bajo contrato con la NASA.

Utiliza también el catálogo Quaia (Storey-Fisher et al. 2024), disponible en Zenodo.

Algunos resultados se obtuvieron con HEALPix (Górski et al. 2005) y healpy (Zonca et
al. 2019).

Referencias de contexto para el resultado: Secrest et al. 2021 (ApJL 908, L51);
Dam, Lewis & Brewer 2023; *Colloquium: The Cosmic Dipole Anomaly* (arXiv:2505.23526).

---

## Nota sobre la elaboración

El análisis fue dirigido y ejecutado por el autor. El código se escribió con
asistencia de modelos de lenguaje, y varias de las decisiones metodológicas surgieron
de esa interacción. Todos los resultados fueron ejecutados y verificados sobre datos
reales por el autor. Se hace explícito porque es información relevante para quien
evalúe el trabajo.

---

## Licencia

Código bajo licencia MIT. Documento e informe bajo CC BY 4.0.
