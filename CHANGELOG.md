# Changelog — CFO Simulation Tool (v3, consolidado)

Este paquete fusiona las dos líneas de trabajo que iban por separado:
(1) tu código, con todos los bugs ya corregidos ronda a ronda, y
(2) las 5 mejoras del "Bloque 2" que se habían construido sobre una versión anterior.
Todo verificado: compila, arranca (`streamlit run app.py`), pasa `ruff` sin
avisos críticos (F821/F401), y pasa los 23 tests de `pytest`.

## Bugs corregidos (acumulado de todas las rondas)
- `run_scenario_module()` — ya no exige argumentos que nadie pasaba.
- `ticker` indefinido en la ruta de subida de CSV (Forecasting).
- Typos `susbet`→`subset` (Prophet) y `apend`→`append` (XGBoost).
- Historial 2D sin aplanar en `forecast_xgboost`.
- `float(booster.predict(dtest))` — incompatible con NumPy 2.x, corregido a `[0]`.
- `_call_openai`: firma y llamada ya no desalineadas; ya no ignora la API key escrita en la UI.
- Typo `lendth`→`length`, y `length` ahora es un parámetro real de `_call_openai`.
- Tabla de KPIs conectada de verdad al prompt (`kpi_summary` ya no es decorativo).
- Resultados de Forecasting ya no ocultos dentro del expander "CFO Manual Forecast".
- Tornado de sensibilidad: ahora es un shock relativo real, no absoluto con etiqueta engañosa.
- CapEx, D&A y ΔNWC del FCF escalan correctamente año a año (antes usaban un valor fijo).
- Imports sin usar eliminados vía `ruff --fix`.

## Mejoras del Bloque 2 (ya integradas sobre el código corregido)
- **Puente entre módulos** (`Modules/shared/state_bridge.py`): Forecasting y Scenario Planning guardan sus resultados; el módulo de informes los detecta y ofrece usarlos con un clic.
- **Grounding/citación**: cada informe generado exige citar el campo de origen de cada cifra.
- **Traza de auditoría** (`Modules/shared/audit_log.py`): cada acción de IA queda registrada, visible en la barra lateral.
- **"⚡ Analyze & Advise"**: botón de un clic en la home que encadena forecast → escenario pesimista → contexto listo para el informe.
- **Bandas de probabilidad**: intervalo del 90% sobre el mejor modelo (nativo en Prophet, bootstrap sobre residuos para el resto) — cierra la brecha con la promesa de "probabilistic AI modules" del abstract del TFM.

## Ronda final
- `_safe_run` ya no envuelve la página entera en `st.status(...)` — eso colapsaba en cada rerun (cada clic/slider), dando la sensación de que "la página se cierra sola" en Scenario Planning y GenAI Reports.
- Añadido `README.md` con quickstart, comandos de desarrollo y notas de diseño.
- Verificación final: compila, `ruff check .` sin avisos (0 F821/F401), 23/23 tests, arranque real confirmado con `streamlit run`.

## Fix post-entrega #2 — mismo bug, otro punto de impacto
- El fix anterior arreglaba `bootstrap_interval` por dentro, pero `best_preds` en bruto (todavía 2D en algunos entornos) se usaba directamente para construir `band_df` con `pd.DataFrame(...)`, que pandas rechaza con "Per-column arrays must each be 1-dimensional". Fix definitivo: normalización a 1D de **todos** los resultados de modelos justo después de calcularlos (`forecasting_app.py`), en un único punto, en vez de parchear cada sitio donde se usan después. De paso, corregido un `import numpy as np` que faltaba en ese archivo (detectado por `ruff` antes de empaquetar).
- `bootstrap_interval` en `forecasting_models.py`: cuando las predicciones del mejor modelo llegan con forma `(n, 1)` (vector columna) en vez de `(n,)` (según la versión de statsmodels/xgboost del entorno), la resta `y_true - y_pred` hacía *broadcasting* de NumPy y producía sin querer una matriz `(n, n)`, reventando la asignación posterior con `ValueError: could not broadcast...`. Fix: `.ravel()` en las tres entradas al inicio de la función, forzando 1D siempre. Añadidos 4 tests de regresión (`TestBootstrapInterval`), incluido el caso exacto que reportaste.
Nueva funcionalidad, no una mejora de las cinco anteriores: Scenario Planning
ahora calcula EBITDA, cobertura de intereses (EBITDA/Interest) y apalancamiento
(Deuda/EBITDA) año a año, con semáforo de cumplimiento frente a los covenants
típicos de un préstamo bancario. Un escenario puede ser rentable y aun así
incumplir un covenant — algo que ningún prototipo genérico de "IA para CFOs"
comprueba. El resultado se propaga vía `state_bridge` hasta el generador de
informes de IA, que ahora señala explícitamente cualquier incumplimiento
proyectado. Es la funcionalidad que conecta el prototipo directamente con la
combinación de finanzas + regulación + IA del perfil detrás del proyecto,
no una feature de IA genérica más.

## Herramientas de calidad añadidas
- `tests/test_forecasting_models.py` — 23 tests con `pytest`.
- `signature_check.py` — detector de desajustes de firma/llamada (arity).
- `.pre-commit-config.yaml` — ejecuta `ruff` (F821/F401/F841) antes de cada commit.

## Pendiente, no incluido en esta consolidación
- Los 3 `F841` menores en `genai_reports.py` (`base_hint`, `init_length`, `context_text`) — variables calculadas y no usadas, cosmético.
- Decidir si archivar `ai_reporting_app.py` si todavía existe en tu repo (módulo huérfano, no lo usa `app.py`).
