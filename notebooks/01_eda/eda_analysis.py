import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#0f1117",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#333",
    "figure.titlesize": 16,
    "axes.titlesize": 13,
    "axes.titlepad": 12,
})

COLORS = {
    "SOM": "#e74c3c", "SSD": "#e67e22", "ETH": "#f39c12",
    "CAR": "#9b59b6", "COD": "#3498db", "MLI": "#1abc9c",
    "NER": "#2ecc71", "TCD": "#e91e63", "MRT": "#00bcd4",
    "SEN": "#8bc34a"
}

REPORTS_DIR = "reports/figures"
import os
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Load Data ───────────────────────────────────────────────
def load_data():
    df = pd.read_csv("data/processed/merged_dataset.csv", parse_dates=["date"])
    print(f"✅ Dataset loaded: {df.shape}")
    print(f"   Countries: {df['country'].nunique()}")
    print(f"   Period: {df['date'].min()} → {df['date'].max()}")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n📊 Missing values:\n{df.isnull().sum()}")
    return df

# ── Plot 1 : IPC par pays ───────────────────────────────────
def plot_ipc_by_country(df):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("🌍 Food Insecurity (IPC Phase) by Country — 2018 to 2024",
                 fontsize=16, color="white", y=1.02)

    for idx, (country, ax) in enumerate(zip(df["country"].unique(), axes.flatten())):
        country_df = df[df["country"] == country]
        color = COLORS.get(country, "#ffffff")

        ax.fill_between(country_df["date"], country_df["ipc_phase"],
                        alpha=0.3, color=color)
        ax.plot(country_df["date"], country_df["ipc_phase"],
                color=color, linewidth=2)

        # IPC threshold lines
        for phase, label, lc in [(3, "Crisis", "#e74c3c"), (4, "Emergency", "#8e44ad")]:
            ax.axhline(y=phase, color=lc, linestyle="--", alpha=0.5, linewidth=1)

        ax.set_title(country, color=color, fontweight="bold")
        ax.set_ylim(1, 5)
        ax.set_ylabel("IPC Phase" if idx % 5 == 0 else "")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = f"{REPORTS_DIR}/01_ipc_by_country.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"✅ Saved: {path}")
    plt.show()

# ── Plot 2 : Évolution conflits ─────────────────────────────
def plot_conflict_trends(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("⚔️ Conflict Trends — Events & Fatalities", fontsize=16, color="white")

    top_conflict = ["SOM", "SSD", "ETH", "CAR", "COD"]

    for country in top_conflict:
        cdf = df[df["country"] == country]
        color = COLORS[country]
        ax1.plot(cdf["date"], cdf["total_events"],
                 label=country, color=color, linewidth=2, alpha=0.85)
        ax2.plot(cdf["date"], cdf["total_fatalities"],
                 label=country, color=color, linewidth=2, alpha=0.85)

    ax1.set_title("Monthly Conflict Events", color="white")
    ax1.set_ylabel("Number of Events")
    ax1.legend(loc="upper right", framealpha=0.3)
    ax1.grid(True, alpha=0.2)

    ax2.set_title("Monthly Fatalities", color="white")
    ax2.set_ylabel("Fatalities")
    ax2.legend(loc="upper right", framealpha=0.3)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f"{REPORTS_DIR}/02_conflict_trends.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"✅ Saved: {path}")
    plt.show()

# ── Plot 3 : Déplacements ───────────────────────────────────
def plot_displacement(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle("🚶 Population Displacement — IDPs & Refugees", fontsize=16, color="white")

    for country in df["country"].unique():
        cdf = df[df["country"] == country]
        if "total_displaced" in cdf.columns:
            ax.plot(cdf["date"], cdf["total_displaced"] / 1e6,
                    label=country, color=COLORS.get(country, "white"),
                    linewidth=2, alpha=0.85)

    ax.set_ylabel("Displaced Population (Millions)")
    ax.legend(loc="upper left", framealpha=0.3, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f"{REPORTS_DIR}/03_displacement.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"✅ Saved: {path}")
    plt.show()

# ── Plot 4 : Heatmap corrélations ──────────────────────────
def plot_correlation_heatmap(df):
    numeric_cols = ["ipc_phase", "population_affected", "total_events",
                    "total_fatalities", "battles", "violence_civilians",
                    "idps", "refugees", "total_displaced"]

    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle("🔥 Correlation Heatmap — Crisis Indicators", fontsize=16, color="white")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    path = f"{REPORTS_DIR}/04_correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"✅ Saved: {path}")
    plt.show()

# ── Plot 5 : Crisis Severity Index ─────────────────────────
def plot_crisis_severity(df):
    df = df.copy()

    # Normalise chaque indicateur entre 0 et 1
    for col in ["ipc_phase", "total_events", "total_fatalities", "total_displaced"]:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val + 1e-8)

    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    df["crisis_severity_index"] = df[norm_cols].mean(axis=1)

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle("🚨 Crisis Severity Index by Country", fontsize=16, color="white")

    for country in df["country"].unique():
        cdf = df[df["country"] == country]
        ax.plot(cdf["date"], cdf["crisis_severity_index"],
                label=country, color=COLORS.get(country, "white"),
                linewidth=2, alpha=0.85)

    ax.axhline(y=0.6, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5,
               label="High Crisis Threshold")
    ax.set_ylabel("Crisis Severity Index (0-1)")
    ax.legend(loc="upper left", framealpha=0.3, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f"{REPORTS_DIR}/05_crisis_severity_index.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"✅ Saved: {path}")
    plt.show()

    # Sauvegarde le dataset enrichi
    df.to_csv("data/processed/dataset_with_csi.csv", index=False)
    print("✅ Dataset with Crisis Severity Index saved!")
    return df

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  HumaCrisis — Exploratory Data Analysis")
    print("=" * 55)

    df = load_data()
    plot_ipc_by_country(df)
    plot_conflict_trends(df)
    plot_displacement(df)
    plot_correlation_heatmap(df)
    df_enriched = plot_crisis_severity(df)

    print("\n" + "=" * 55)
    print("✅ EDA Complete! All figures saved in reports/figures/")
    print("=" * 55)