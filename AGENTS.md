# Repository Guidelines

## Project Structure & Module Organization
The repository centers on standalone Python scripts and notebooks. The `plot_module/` package holds reusable analyzers (`xrd_analyzer.py`, `tc_analyzer.py`, `UVVIS_analyzer.py`) and related utilities. High-level study-specific scripts live in the root (for example `MA_TC1909_Plot.py`, `MA_Tauc_Avr.py`) alongside experiment notebooks (`*.ipynb`). Generated figures and exports are grouped under `Average_Plots/`, `Result_Average/`, `Images/`, and `Single_Plot/`. Treat raw measurement files as inputs kept outside the repo; temporary intermediates should stay in `Result_Average/` or be ignored.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated environment; match Python ≥3.10.
- `pip install numpy pandas matplotlib scipy pybaselines pymatgen jupyter`: install core dependencies used across analyzers.
- `python plot_module/xrd_analyzer.py`: run the XRD workflow interactively; adjust paths within the script before execution.
- `jupyter lab`: open notebooks such as `MA_TC.ipynb` for exploratory analysis.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and descriptive snake_case identifiers. Module-level constants (e.g., API keys) should be uppercase and loaded from environment variables. Keep plotting helpers inside `plot_module/` and prefer function-based organization over sprawling notebooks. Save notebooks with descriptive experiment dates (`YY_MM_Description.ipynb`) and mirror that pattern for scripts.

## Testing Guidelines
Automated tests are not yet configured. Validate changes by running the relevant analyzer or notebook against a representative dataset and confirming regenerated figures in `Result_Average/`. Keep throwaway checks inside `test.py` and remove once verified. When adding new scripts, include a `if __name__ == "__main__":` section that exercises the primary workflow for manual testing.

## Commit & Pull Request Guidelines
Recent commits favor concise summaries ("Update TC analysis workflows"); keep using imperative, 70-character subjects that mention the affected module (`plot_module`, `MA_Tauc_*`, etc.). For pull requests, supply context, sample before/after plots, and link to measurement files or issues. Note any required API tokens (e.g., Materials Project key) and flag breaking changes to data directory expectations.

## Data & Configuration Tips
Store API credentials in environment variables (`export MAPI_KEY=...`) rather than hardcoding. Paths in scripts are relative to the repository root; prefer `Path` objects for cross-platform consistency, and document any expected external folder layout in script docstrings.
