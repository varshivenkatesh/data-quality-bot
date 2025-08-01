import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

def visualize_report(report: dict, output_name: str = "dq_report.json", output_dir: str = "visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(output_name))[0]

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 14))
    axs = axs.flatten()  # Flatten for easy indexing
    plot_idx = 0

    # Nulls
    if 'null_percentage' in report:
        nulls = pd.Series(report['null_percentage']).sort_values(ascending=False)
        nulls_df = pd.DataFrame({'column': nulls.index, 'null_pct': nulls.values})
        sns.barplot(data=nulls_df, x='null_pct', y='column', hue='column', palette='magma', legend=False, ax=axs[plot_idx])
        axs[plot_idx].set_title("Null Values by Column")
        axs[plot_idx].set_xlabel("Null Percentage")
        plot_idx += 1

    # Duplicates
    if 'duplicate_rows_percent' in report:
        axs[plot_idx].bar(['Duplicates', 'Non-Duplicates'],
                          [report['duplicate_rows_percent'], 100 - report['duplicate_rows_percent']],
                          color=['red', 'green'])
        axs[plot_idx].set_title("Duplicate Row Share")
        axs[plot_idx].set_ylabel("%")
        plot_idx += 1

    # Outliers
    if 'outliers_count' in report:
        outliers = pd.Series(report['outliers_count']).sort_values(ascending=False)
        outliers_df = pd.DataFrame({'column': outliers.index, 'count': outliers.values})
        sns.barplot(data=outliers_df, x='count', y='column', hue='column', palette='coolwarm', legend=False, ax=axs[plot_idx])
        axs[plot_idx].set_title("Outliers by Numeric Column")
        axs[plot_idx].set_xlabel("Outlier Count")
        plot_idx += 1

    # Constant Columns — just show names
    if 'columns_with_constant_values' in report:
        const_cols = report['columns_with_constant_values']
        text = "\n".join(const_cols) if const_cols else "None"
        axs[plot_idx].text(0.1, 0.5, text, fontsize=12)
        axs[plot_idx].set_title("Columns with Constant Values")
        axs[plot_idx].axis('off')
        plot_idx += 1

    # High Cardinality — list style
    if 'columns_with_high_cardinality' in report:
        high_card_cols = report['columns_with_high_cardinality']
        text = "\n".join(high_card_cols) if high_card_cols else "None"
        axs[plot_idx].text(0.1, 0.5, text, fontsize=12)
        axs[plot_idx].set_title("High Cardinality Columns")
        axs[plot_idx].axis('off')
        plot_idx += 1

    # Potential Primary Keys — list style
    if 'potential_primary_keys' in report:
        pk_cols = report['potential_primary_keys']
        text = "\n".join(pk_cols) if pk_cols else "None"
        axs[plot_idx].text(0.1, 0.5, text, fontsize=12)
        axs[plot_idx].set_title("Potential Primary Key Columns")
        axs[plot_idx].axis('off')
        plot_idx += 1

    # Turn off any remaining axes
    for ax in axs[plot_idx:]:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{base_name}_visualization.png"))
    plt.close(fig)
