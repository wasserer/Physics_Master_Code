import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


URBACH_SOURCE = Path(
    ""  # Enter your file path/folder path in this place
    "Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/"
    "TC_1909_IV/TC_log/urbach_metrics_test.csv"
)


def _urbach_outlier_mask(values: pd.Series) -> pd.Series:
    """
    Identify inliers in a series using the IQR rule (1.5 * IQR outside fences is an outlier).
    Returns a boolean Series aligned with `values` where True denotes an inlier.
    """
    values = values.dropna()
    if values.empty:
        return pd.Series(dtype=bool)

    if values.size < 4:
        # Too few points for a reliable IQR-based filter -> keep all measurements.
        return pd.Series(True, index=values.index)

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(True, index=values.index)

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (values >= lower) & (values <= upper)
    return mask


def compute_cycle_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group measurements by cycle, remove outliers, then compute summary statistics for Urbach energy.

    Returns a dataframe with columns:
    cycle, urbach_energy_mean, urbach_energy_std, urbach_energy_sem,
    count_raw, count_filtered, count_removed
    """
    required_columns = {"cycle", "urbach_slope"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in input CSV: {', '.join(sorted(missing))}")

    rows = []
    for cycle, group_df in df.groupby("cycle"):
        # Convert slopes to energies via inverse.
        slopes = group_df["urbach_slope"]
        energies = 1.0 / slopes  # Simple reciprocal without zero handling
        mask = _urbach_outlier_mask(energies)

        # Align mask with original group index; exposures dropped in _urbach_outlier_mask are NaNs.
        mask = mask.reindex(group_df.index, fill_value=False)
        filtered = energies[mask]

        count_raw = int(energies.count())
        count_filtered = int(filtered.count())
        count_removed = count_raw - count_filtered

        if count_filtered > 1:
            std_val = float(filtered.std(ddof=1))
            sem_val = std_val / np.sqrt(count_filtered)
        elif count_filtered == 1:
            std_val = 0.0
            sem_val = 0.0
        else:
            std_val = np.nan
            sem_val = np.nan

        mean_val = float(filtered.mean()) if count_filtered else np.nan

        rows.append(
            {
                "cycle": cycle,
                "urbach_energy_mean": mean_val,
                "urbach_energy_std": std_val,
                "urbach_energy_sem": sem_val,
                "count_raw": count_raw,
                "count_filtered": count_filtered,
                "count_removed": count_removed,
            }
        )

    stats = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    return stats


def add_logarithmic_fit(stats: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float] | None]:
    """
    Fit urbach_energy_mean against ln(cycle) for cycles > 0 and add predictions.
    Returns the updated DataFrame and the (a, b) coefficients with y = a * ln(x) + b.
    """
    stats = stats.copy()
    if "urbach_energy_mean" not in stats:
        stats["log_fit_value"] = np.nan
        return stats, None

    valid = stats[np.isfinite(stats["urbach_energy_mean"]) & (stats["cycle"] > 0)]
    if len(valid) < 2:
        stats["log_fit_value"] = np.nan
        return stats, None

    log_x = np.log(valid["cycle"].astype(float).to_numpy())
    y = valid["urbach_energy_mean"].astype(float).to_numpy()
    a, b = np.polyfit(log_x, y, 1)

    stats["log_fit_value"] = np.nan
    positive_mask = stats["cycle"] > 0
    stats.loc[positive_mask, "log_fit_value"] = (
        a * np.log(stats.loc[positive_mask, "cycle"].astype(float)) + b
    )
    return stats, (float(a), float(b))


def save_cycle_statistics(stats: pd.DataFrame, destination: Path) -> Path:
    output_csv = destination.with_name("urbach_energy_cycle_stats.csv")
    stats.to_csv(output_csv, index=False)
    return output_csv


def plot_cycle_statistics(
    stats: pd.DataFrame, destination: Path, fit_params: tuple[float, float] | None
) -> Path:
    output_png = destination.with_name("urbach_energy_cycle_plot.png")

    plot_data = stats.dropna(subset=["urbach_energy_mean"]).sort_values("cycle")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.errorbar(
        plot_data["cycle"],
        plot_data["urbach_energy_mean"],
        yerr=plot_data["urbach_energy_std"],
        fmt="o",
        capsize=4,
        color="#005C53",
        ecolor="#BBBBBB",
        label="Urbach energy (outliers removed)",
    )

    if "log_fit_value" in stats.columns:
        fit_data = stats.dropna(subset=["log_fit_value"]).sort_values("cycle")
        if not fit_data.empty:
            if fit_params is not None:
                a, b = fit_params
                label = f"Log fit (y = {a:.4g} ln(x) + {b:.4g})"
            else:
                label = "Log fit"
            ax.plot(
                fit_data["cycle"],
                fit_data["log_fit_value"],
                color="#5993BD",
                linewidth=2,
                label=label,
            )

    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Urbach energy")
    #ax.set_title("Urbach energy vs. cycle number (outlier-trimmed)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    return output_png

def plot_cycle_statistics_fill(
    stats: pd.DataFrame, destination: Path, fit_params: tuple[float, float] | None
) -> Path:
    output_png = destination.with_name("urbach_energy_cycle_plot.png")

    # 清理和排序
    plot_data = stats.dropna(subset=["urbach_energy_mean"]).sort_values("cycle")

    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 5))

    # 提取数据
    x = plot_data["cycle"]
    y = plot_data["urbach_energy_mean"]
    y_err = plot_data["urbach_energy_std"]

    # 绘制主曲线
    ax.plot(
        x,
        y,
        color="#005C53",
        linewidth=2.5,
        label="Urbach energy (outliers removed)",
    )

    # 使用 fill_between 绘制误差阴影带
    ax.fill_between(
        x,
        y - y_err,
        y + y_err,
        color="#005C53",
        alpha=0.25,
        linewidth=0,
    )

    # 如果存在 log 拟合数据，则绘制拟合曲线
    if "log_fit_value" in stats.columns:
        fit_data = stats.dropna(subset=["log_fit_value"]).sort_values("cycle")
        if not fit_data.empty:
            if fit_params is not None:
                a, b = fit_params
                label = f"Log fit (y = {a:.4g} ln(x) + {b:.4g})"
            else:
                label = "Log fit"
            ax.plot(
                fit_data["cycle"],
                fit_data["log_fit_value"],
                color="#5993BD",
                linewidth=2,
                linestyle='--',
                label=label,
            )

    # 坐标轴与网格
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Urbach energy")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 布局与保存
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    return output_png



def main() -> None:
    if not URBACH_SOURCE.exists():
        raise FileNotFoundError(f"Could not find input CSV at {URBACH_SOURCE}")

    urbach_df = pd.read_csv(URBACH_SOURCE)
    cycle_stats = compute_cycle_statistics(urbach_df)
    cycle_stats, fit_params = add_logarithmic_fit(cycle_stats)

    stats_csv = save_cycle_statistics(cycle_stats, URBACH_SOURCE)
    #plot_png = plot_cycle_statistics(cycle_stats, URBACH_SOURCE, fit_params)
    plot_png = plot_cycle_statistics_fill(cycle_stats, URBACH_SOURCE, fit_params)

    for _, row in cycle_stats.iterrows():
        cycle = int(row["cycle"])
        mean_val = row["urbach_energy_mean"]
        std_val = row["urbach_energy_std"]
        sem_val = row["urbach_energy_sem"]
        print(
            f"Cycle {cycle}: mean={mean_val:.6f} std={std_val:.6f} "
            f"sem={sem_val:.6f} (kept {int(row['count_filtered'])} / removed {int(row['count_removed'])})"
        )

    print(f"\nWrote cycle statistics to: {stats_csv}")
    print(f"Saved plot with error bars to: {plot_png}")
    if fit_params is not None:
        a, b = fit_params
        print(f"Logarithmic fit coefficients: a={a:.6f}, b={b:.6f}")


if __name__ == "__main__":
    main()
