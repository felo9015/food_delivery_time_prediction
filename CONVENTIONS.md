# Repo conventions

This project separates the **working process** from the **final deliverable**:

- The `*_exploration.ipynb` notebooks (`sql/sql_exploration.ipynb`, `notebooks/eda_exploration.ipynb`, `notebooks/model_exploration.ipynb`, `api/api_exploration.ipynb`) are the trial-and-error workspace: this is where exploration happens, hypotheses are tested, alternatives are compared, and reasoning is documented step by step.
- `sql/sql_queries.sql`, `model_pipeline/`, `api/main.py`, and the `.md` reports (`EDA_report.md`, `model_notes.md`, `explainability.md`, `error_insights.md`, `strategic_reflections.md`, `sql/sql_insights.md`) are the final deliverables: clean, polished, documented code or conclusions, free of exploratory noise.

This separation is intentional, not an oversight: it shows the full analysis and decision-making process without cluttering the formal deliverable being evaluated.
