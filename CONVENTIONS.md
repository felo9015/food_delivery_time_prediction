# Convenciones del repo

Este proyecto separa el **proceso de trabajo** del **entregable final**:

- Los notebooks `*_exploration.ipynb` (`sql/sql_exploration.ipynb`, `notebooks/eda_exploration.ipynb`, `notebooks/model_exploration.ipynb`, `api/api_exploration.ipynb`) son el espacio de prueba y error: ahí se explora, se prueban hipótesis, se comparan alternativas y se documenta el razonamiento paso a paso.
- `sql/sql_queries.sql`, `model_pipeline/`, `api/main.py`, y los reportes en `.md` (`EDA_report.md`, `model_notes.md`, `explainability.md`, `error_insights.md`, `strategic_reflections.md`, `sql/sql_insights.md`) son los entregables finales: código o conclusiones ya limpios, pulidos y documentados, sin el ruido del proceso exploratorio.

Esta separación es intencional, no descuido: permite mostrar el proceso completo de análisis y decisión sin ensuciar el entregable formal que se evalúa.
