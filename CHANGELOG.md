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
- (Las bandas de probabilidad quedan como siguiente paso — ver más abajo.)

## Herramientas de calidad añadidas
- `tests/test_forecasting_models.py` — 23 tests con `pytest`.
- `signature_check.py` — detector de desajustes de firma/llamada (arity).
- `.pre-commit-config.yaml` — ejecuta `ruff` (F821/F401/F841) antes de cada commit.

## Pendiente, no incluido en esta consolidación
- Bandas de probabilidad del 90% en el módulo de Forecasting (estaban en el zip anterior del Bloque 2; no se han vuelto a cablear en esta versión — avisa si las quieres de vuelta).
- Los 3 `F841` menores en `genai_reports.py` (`base_hint`, `init_length`, `context_text`) — variables calculadas y no usadas, cosmético.
- Decidir si archivar `ai_reporting_app.py` si todavía existe en tu repo (módulo huérfano, no lo usa `app.py`).
