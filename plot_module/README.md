#Important: this is a old version of readme, and it is an on-going project. New docs will be updated later.
# Plot Module

This module provides a unified interface to plot different types of data files including XRD patterns, IV curves, IV sweeps, and summary analysis plots.

## Installation

No installation required. Just import the module or run the scripts.

## Usage

Use the `import_and_plot(file_path, mode)` function from this module to plot different file types.

### Modes
- `xrd`: for XRD data (two-column text: angle, intensity)
- `iv`: for IV curves (two-column text: voltage, current)
- `iv_sweep`: for IV sweep data (CSV: voltage, current density)
- `summary`: for multi-series analysis (CSV with header)

### Example

```python
from plot_module import import_and_plot

import_and_plot("your_data_file.xy", mode="xrd")
import_and_plot("your_iv_data.txt", mode="iv")
import_and_plot("your_sweep_data.csv", mode="iv_sweep")
import_and_plot("your_summary_data.csv", mode="summary")
```

Make sure the file format matches the expected input for each mode.
