import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


TAUC_SOURCE = Path(
    "/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/"
    "Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/"
    "TC_1909_IV/TC_log/tauc_metrics_test.csv"
)


def _outlier_mask(values: pd.Series) -> pd.Series:
    """Return a boolean mask marking inliers using the 1.5*IQR rule."""
    values = values.dropna()
    if values.empty:
        return pd.Series(dtype=bool)

    if values.size < 4:
        return pd.Series(True, index=values.index)

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(True, index=values.index)

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (values >= lower) & (values <= upper)


def compute_cycle_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers per cycle and compute summary stats of the Tauc band gap.

    Produces columns:
    cycle, band_gap_mean, band_gap_std, band_gap_sem,
    count_raw, count_filtered, count_removed
    """
    required_columns = {"cycle", "band_gap"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in input CSV: {', '.join(sorted(missing))}")

    rows = []
    for cycle, group_df in df.groupby("cycle"):
        band_gaps = group_df["band_gap"]
        mask = _outlier_mask(band_gaps)
        mask = mask.reindex(group_df.index, fill_value=False)
        filtered = band_gaps[mask]

        count_raw = int(band_gaps.count())
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
                "band_gap_mean": mean_val,
                "band_gap_std": std_val,
                "band_gap_sem": sem_val,
                "count_raw": count_raw,
                "count_filtered": count_filtered,
                "count_removed": count_removed,
            }
        )

    stats = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    return stats


def save_cycle_statistics(stats: pd.DataFrame, destination: Path) -> Path:
    output_csv = destination.with_name("tauc_band_gap_cycle_stats.csv")
    stats.to_csv(output_csv, index=False)
    return output_csv


def plot_cycle_statistics(stats: pd.DataFrame, destination: Path) -> Path:
    output_png = destination.with_name("tauc_band_gap_cycle_plot.png")

    plot_data = stats.dropna(subset=["band_gap_mean"]).sort_values("cycle")
    fig, ax = plt.subplots(figsize=(8, 5))
    #Use the "fill between" to plot the 
    #Plot the errorbar:
    ax.errorbar(
        plot_data["cycle"],
        plot_data["band_gap_mean"],
        yerr=plot_data["band_gap_std"],
        fmt="o",
        capsize=4,
        color="#C95A49",
        ecolor="#BBBBBB",
        label="Band gap (outliers removed)",
    )

    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Band gap (eV)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    return output_png

def plot_cycle_stats_fill_between(stats: pd.DataFrame, destination: Path) -> Path:
    output_png = destination.with_name("tauc_band_gap_cycle_plot_fill_between.png")

    # 清理并按循环数排序
    plot_data = stats.dropna(subset=["band_gap_mean"]).sort_values("cycle")

    # 创建图
    fig, ax = plt.subplots(figsize=(8, 5))

    # 提取数据
    x = plot_data["cycle"]
    y = plot_data["band_gap_mean"]
    y_err = plot_data["band_gap_std"]

    # 主曲线
    ax.plot(
        x,
        y,
        color="#C95A49",
        linewidth=2.5,
        label="Band gap",
    )

    # 使用 fill_between 添加误差阴影带
    ax.fill_between(
        x,
        y - y_err,
        y + y_err,
        color="#C95A49",
        alpha=0.25,   # 阴影透明度
        linewidth=0,  # 不需要边框线
    )

    # 坐标轴标签和样式
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Band gap (eV)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 布局与保存
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

def main() -> None:
    if not TAUC_SOURCE.exists():
        raise FileNotFoundError(f"Could not find input CSV at {TAUC_SOURCE}")

    tauc_df = pd.read_csv(TAUC_SOURCE)
    cycle_stats = compute_cycle_statistics(tauc_df)

    stats_csv = save_cycle_statistics(cycle_stats, TAUC_SOURCE)
    #plot_png = plot_cycle_statistics(cycle_stats, TAUC_SOURCE)
    plot_png = plot_cycle_stats_fill_between(cycle_stats, TAUC_SOURCE)

    for _, row in cycle_stats.iterrows():
        cycle = int(row["cycle"])
        mean_val = row["band_gap_mean"]
        std_val = row["band_gap_std"]
        sem_val = row["band_gap_sem"]
        print(
            f"Cycle {cycle}: mean={mean_val:.6f} std={std_val:.6f} "
            f"sem={sem_val:.6f} (kept {int(row['count_filtered'])} / removed {int(row['count_removed'])})"
        )

    print(f"\nWrote cycle statistics to: {stats_csv}")
    print(f"Saved plot with error bars to: {plot_png}")


if __name__ == "__main__":
    main()
