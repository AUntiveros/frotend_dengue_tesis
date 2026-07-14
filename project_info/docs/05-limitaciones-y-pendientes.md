# Limitaciones y pendientes

## Limitaciones actuales (honestas para la tesis)

1. **Solo H4 entrenado.** Los artefactos guardados predicen a 4 semanas. La UI
   deshabilita H1/H8/H12/H24. La metadata lista `horizontes=[4,8,12]` y existen
   `target_h8/target_h12` en la AMT → H8/H12 son reentrenables (ver pendientes).

2. **Variantes ZIP M7a/c/d/e: solo comparativa.** Solo M7b guardó los arrays de
   inferencia rápida (`zip_beta_mean/intercept/psi_posterior`). Las demás están
   en traces `.nc` que requieren PyMC/ArviZ y rutas más complejas (splines,
   offset XGBoost). Aparecen en el selector con sus métricas pero sin mapa.

3. **LightGBM/XGBoost/RF "global": descartados.** Se entrenaron con subconjuntos
   de 20-30 features incompatibles con las 41 de `all_features`. El generador
   los omite por validación de `n_features`.

4. **Prophet es nacional, no distrital.** Su `.pkl` modela la serie agregada del
   país; métricas pobres (R²=-0.042). Se muestra como serie nacional.

5. **Importancia de variables.**
   - ZIP: contribución local real β·x al log-λ (defendible).
   - Árboles: importancia **global** (misma para todos los distritos).
   - BiLSTM: sin importancia nativa (aviso "en proceso").
   - SHAP/LIME por distrito y análisis DIC por variable: **pendientes**.

6. **Predicción = forecast del modelo en la semana de referencia.** La AMT real
   llega hasta 2026-03-09; lo posterior es relleno cero. El "H4" mostrado es el
   forecast para ref+4 a partir de las features de la semana de referencia.

## Pendientes / decisiones abiertas

- [ ] **Reentrenar ZIP H8/H12** localmente desde la AMT para 3 horizontes reales
      (existen `target_h8`, `target_h12`). Habilitaría más botones de horizonte.
- [ ] **SHAP/LIME real** por distrito para los árboles, y análisis **DIC** del
      aporte de cada variable explicativa (actualmente importancia global).
- [ ] **Exportar arrays de inferencia rápida de M7a/c/d/e** desde sus traces
      (requiere PyMC/ArviZ) para sumarlas al mapa.
- [ ] **Reducir tamaño del JSON** (12 MB) si pesa: redondear `series`/`hist` a
      enteros, o separar historia observada (un fetch perezoso aparte).
- [ ] **Habilitar gzip/brotli** en el servidor de producción para el JSON.

## Re-generación semanal

Al actualizar `AMT_final.parquet` con nuevos casos:
```bash
cd backend && python generador_dashboard.py
```
La semana de referencia se recalcula sola (última con casos > 0).
