"""Simple GUI to analyze a single IV .dat file and plot the curve."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from plot_module.solsim_analyzer_V2 import solarSimulator


class IVPlotGUI:
    """Tkinter based GUI for solar simulator single-file analysis."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("MA Thermal Cycling IV Analyzer")
        self.root.geometry("1050x650")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.simulator = solarSimulator()
        self.file_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a .dat file to begin.")

        self.param_vars: dict[str, tk.StringVar] = {}
        self.figure = Figure(figsize=(6.5, 4.5), dpi=100)
        self.canvas: FigureCanvasTkAgg | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        file_frame = ttk.Frame(self.root, padding=10)
        file_frame.grid(row=0, column=0, sticky="ew")
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Measurement file:").grid(row=0, column=0, padx=(0, 8))
        ttk.Entry(file_frame, textvariable=self.file_path_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(file_frame, text="Browse…", command=self.select_file).grid(row=0, column=2, padx=8)
        ttk.Button(file_frame, text="Load", command=self.load_current_file).grid(row=0, column=3)

        content = ttk.Frame(self.root, padding=(10, 5))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        params_frame = ttk.Labelframe(content, text="Photovoltaic parameters", padding=10)
        params_frame.grid(row=0, column=0, sticky="ns")

        labels = [
            "Voc (V)",
            "Isc (mA/cm²)",
            "V_MPP (V)",
            "I_MPP (mA/cm²)",
            "FF",
            "PCE (%)",
            "Rs (Ω)",
            "Rp (Ω)",
            "Pixel area (cm²)",
            "Intensity (mW/cm²)",
        ]
        for idx, name in enumerate(labels):
            ttk.Label(params_frame, text=name + ":").grid(row=idx, column=0, sticky="w", pady=2, padx=(0, 6))
            var = tk.StringVar(value="—")
            ttk.Label(params_frame, textvariable=var, width=18).grid(row=idx, column=1, sticky="e")
            self.param_vars[name] = var

        plot_container = ttk.Labelframe(content, text="IV curve", padding=10)
        plot_container.grid(row=0, column=1, sticky="nsew", padx=(15, 0))
        plot_container.rowconfigure(0, weight=1)
        plot_container.columnconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.figure.subplots_adjust(bottom=0.15, left=0.15)

        status_frame = ttk.Frame(self.root, padding=10)
        status_frame.grid(row=2, column=0, sticky="ew")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w")

    def select_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select IV .dat file",
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")],
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.load_current_file()

    def load_current_file(self) -> None:
        raw_path = self.file_path_var.get().strip()
        if not raw_path:
            messagebox.showinfo("No file", "Please choose a .dat file.")
            return

        path = Path(raw_path).expanduser()
        if not path.exists():
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return

        self.status_var.set(f"Loading {path.name} …")
        self.root.update_idletasks()

        try:
            self.simulator.filePath = str(path)
            self.simulator.loadFileData_Cycling()
        except Exception as exc:
            messagebox.showerror("Load error", f"Failed to analyze file:\n{exc}")
            self.status_var.set("Error while loading file.")
            return

        self._update_params()
        self._draw_plot(path.name)
        self.status_var.set(f"Loaded {path.name}")

    def _update_params(self) -> None:
        pixel_area = 1000.0 / self.simulator.CDC if self.simulator.CDC else np.nan
        entries = {
            "Voc (V)": self.simulator.Voc,
            "Isc (mA/cm²)": self.simulator.Isc,
            "V_MPP (V)": self.simulator.V_MPP,
            "I_MPP (mA/cm²)": self.simulator.I_MPP,
            "FF": self.simulator.FF,
            "PCE (%)": self.simulator.PCE,
            "Rs (Ω)": self.simulator.Rs,
            "Rp (Ω)": self.simulator.Rp,
            "Pixel area (cm²)": pixel_area,
            "Intensity (mW/cm²)": self.simulator.intensity,
        }
        for label, value in entries.items():
            var = self.param_vars.get(label)
            if not var:
                continue
            var.set(self._format_number(value, label))

    def _draw_plot(self, label: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        voltages = np.asarray(self.simulator.voltages, dtype=float)
        currents = np.asarray(self.simulator.currents, dtype=float)

        if voltages.size == 0 or currents.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            ax.plot(voltages, currents, label=label)
            if self._is_valid_number(self.simulator.V_MPP) and self._is_valid_number(self.simulator.I_MPP):
                ax.scatter(
                    [self.simulator.V_MPP],
                    [self.simulator.I_MPP],
                    color="tab:red",
                    label="MPP",
                    zorder=5,
                )

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current density (mA/cm²)")
        ax.set_xlim(-1, 0)
        ax.set_ylim(-1, 8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        self.canvas.draw()

    def _format_number(self, value: float | None, label: str) -> str:
        if value is None:
            return "—"
        try:
            if isinstance(value, (float, int, np.floating, np.integer)):
                if np.isnan(value):
                    return "—"
                precision = 3 if label not in {"FF", "PCE (%)"} else 2
                return f"{value:.{precision}f}"
        except Exception:
            pass
        return str(value)

    def _is_valid_number(self, value: float | None) -> bool:
        if value is None:
            return False
        try:
            return not np.isnan(value)
        except Exception:
            return False

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    gui = IVPlotGUI()
    gui.run()
