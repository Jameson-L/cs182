import pandas as pd
import matplotlib.pyplot as plt

# Configuration
CSV_PATHS = [
    "in/41.csv",
    "in/43.csv",
    "in/45.csv",
    "in/47.csv",
    "in/49.csv",
]
SAVE_PLOTS = True
OUT_DIR = "./out"

# Load and prepare data
dfs = []
baseline_dfs = []
for csv_path in CSV_PATHS:
    df = pd.read_csv(csv_path)
    baseline_mask = (df["pca"].astype(str).str.lower() == "none") | (df["pca"].isna())
    baseline_df = df[baseline_mask].copy()
    regular_df = df[~baseline_mask].copy()
    
    regular_df["pca"] = pd.to_numeric(regular_df["pca"], errors="coerce").astype(int)
    regular_df["layer"] = pd.to_numeric(regular_df["layer"], errors="coerce").astype(int)
    baseline_df["layer"] = pd.to_numeric(baseline_df["layer"], errors="coerce").astype(int)
    baseline_df["pca"] = None
    
    dfs.append(regular_df)
    baseline_dfs.append(baseline_df)

combined_df = pd.concat(dfs, ignore_index=True)
baseline_combined_df = pd.concat(baseline_dfs, ignore_index=True)

# Define metrics to plot
metrics = {
    "val_loss": "Validation Loss",
    "rouge1": "ROUGE-1",
    "rougeL": "ROUGE-L",
    "sentencebert": "Sentence-BERT",
    "bleu": "BLEU Score",
}

metrics = {k: v for k, v in metrics.items() if k in combined_df.columns}

# Compute medians and save to CSV
metric_columns = [col for col in combined_df.columns 
                  if col not in ['layer', 'pca'] and pd.api.types.is_numeric_dtype(combined_df[col])]

median_df = combined_df.groupby(['layer', 'pca'])[metric_columns].median().reset_index()
baseline_median_df = baseline_combined_df.groupby('layer')[metric_columns].median().reset_index()
baseline_median_df['pca'] = None

median_df = pd.concat([baseline_median_df, median_df], ignore_index=True)
median_df = median_df.sort_values(['layer', 'pca'], na_position='first')
median_df['pca'] = median_df['pca'].apply(lambda x: 'None' if pd.isna(x) or x is None else str(int(x)))

column_order = ['layer', 'pca'] + metric_columns
median_df = median_df[column_order]

median_csv_path = f"{OUT_DIR}/medians.csv"
median_df.to_csv(median_csv_path, index=False)
print(f"Saved medians to {median_csv_path}")

# Prepare baseline statistics for plotting
baseline_medians = {}
for layer in baseline_combined_df['layer'].unique():
    baseline_medians[layer] = {}
    layer_baseline_data = baseline_combined_df[baseline_combined_df['layer'] == layer]
    for metric in metric_columns:
        baseline_medians[layer][metric] = {
            'median': layer_baseline_data[metric].median(),
            'q1': layer_baseline_data[metric].quantile(0.25),
            'q3': layer_baseline_data[metric].quantile(0.75)
        }

layers = sorted(combined_df["layer"].unique())
cmap = plt.get_cmap("tab10")
colors = {layer: cmap(i % cmap.N) for i, layer in enumerate(layers)}

# Generate plots
for metric_key, metric_label in metrics.items():
    plt.figure(figsize=(8, 5))
    for layer in layers:
        layer_data = combined_df[combined_df["layer"] == layer]
        if layer_data.empty:
            continue
        
        grouped = layer_data.groupby("pca")[metric_key].agg([
            ('median', 'median'),
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75))
        ]).reset_index()
        grouped = grouped.sort_values("pca")
        
        yerr_lower = grouped["median"] - grouped["q1"]
        yerr_upper = grouped["q3"] - grouped["median"]
        
        plt.errorbar(
            grouped["pca"],
            grouped["median"],
            yerr=[yerr_lower, yerr_upper],
            marker="o",
            label=f"Layer {layer}",
            color=colors[layer],
            capsize=2,
            capthick=0.5,
            linewidth=0.8,
            markersize=3
        )
        
        if layer in baseline_medians and metric_key in baseline_medians[layer]:
            baseline_stats = baseline_medians[layer][metric_key]
            baseline_median = baseline_stats['median']
            baseline_q1 = baseline_stats['q1']
            baseline_q3 = baseline_stats['q3']
            
            plt.axhline(y=baseline_q1, color=colors[layer], linestyle='--', linewidth=0.5, alpha=0.3)
            plt.axhline(y=baseline_q3, color=colors[layer], linestyle='--', linewidth=0.5, alpha=0.3)
            plt.axhline(y=baseline_median, color=colors[layer], linestyle='--', linewidth=1, alpha=0.7)

    plt.title(f"{metric_label} vs PCA Components (Median with IQR)")
    plt.xlabel("PCA Components")
    plt.ylabel(metric_label)
    plt.grid(True)
    plt.legend(title="Layers")
    plt.tight_layout()

    if SAVE_PLOTS:
        fname = f"{OUT_DIR}/{metric_key}_vs_pca_median.png"
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
