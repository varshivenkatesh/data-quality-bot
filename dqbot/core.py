import json
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# If using YAML configs
try:
    import yaml
except ImportError:
    yaml = None
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

def load_config(path: str) -> dict:
    """
    Load JSON or YAML configuration file specifying custom checks.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        if path.lower().endswith(('.yaml', '.yml')):
            if yaml is None:
                raise ImportError("PyYAML is required to load YAML configs. Install with `pip install pyyaml`.")
            return yaml.safe_load(f)
        else:
            return json.load(f)


def get_default_config() -> dict:
    """
    Return default configuration with all available metrics enabled.
    """
    return {
        'basic_stats': {
            'enabled': True,
            'include': ['num_rows', 'num_columns', 'column_types', 'columns']
        },
        'null_analysis': {
            'enabled': True,
            'include_counts': True,
            'include_percentages': True
        },
        'duplicate_analysis': {
            'enabled': True,
            'include_count': True,
            'include_percentage': True
        },
        'unique_values': {
            'enabled': True
        },
        'constant_values': {
            'enabled': True
        },
        'high_cardinality': {
            'enabled': True,
            'threshold': 0.5  # Fraction of total rows
        },
        'primary_key_detection': {
            'enabled': True
        },
        'mixed_types': {
            'enabled': True
        },
        'whitespace_issues': {
            'enabled': True
        },
        'outlier_detection': {
            'enabled': True,
            'method': 'iqr',  # or 'zscore'
            'iqr_multiplier': 1.5,
            'zscore_threshold': 3
        },
        'value_distribution': {
            'enabled': True,
            'top_n': 5
        },
        'date_analysis': {
            'enabled': True,
            'check_future_dates': True
        },
        'custom_checks': {
            'numeric_ranges': [],
            'date_validation': [],
            'value_sets': []
        }
    }


def basic_stats_check(df: pd.DataFrame, config: dict) -> dict:
    """Basic dataset statistics."""
    if not config.get('basic_stats', {}).get('enabled', True):
        return {}
    
    include = config.get('basic_stats', {}).get('include', ['num_rows', 'num_columns', 'column_types', 'columns'])
    result = {}
    
    if 'num_rows' in include:
        result['num_rows'] = len(df)
    if 'num_columns' in include:
        result['num_columns'] = len(df.columns)
    if 'column_types' in include:
        result['column_types'] = df.dtypes.astype(str).to_dict()
    if 'columns' in include:
        result['columns'] = df.columns.tolist()
    
    return result


def null_analysis_check(df: pd.DataFrame, config: dict) -> dict:
    """Null value analysis."""
    if not config.get('null_analysis', {}).get('enabled', True):
        return {}
    
    result = {}
    null_config = config.get('null_analysis', {})
    
    if null_config.get('include_counts', True):
        result['null_counts'] = df.isnull().sum().to_dict()
    
    if null_config.get('include_percentages', True):
        result['null_percentage'] = (df.isnull().sum() / max(len(df), 1) * 100).round(2).to_dict()
    
    return result


def duplicate_analysis_check(df: pd.DataFrame, config: dict) -> dict:
    """Duplicate row analysis."""
    if not config.get('duplicate_analysis', {}).get('enabled', True):
        return {}
    
    result = {}
    dup_config = config.get('duplicate_analysis', {})
    duplicate_count = int(df.duplicated().sum())
    
    if dup_config.get('include_count', True):
        result['duplicate_row_count'] = duplicate_count
    
    if dup_config.get('include_percentage', True):
        result['duplicate_rows_percent'] = round(duplicate_count / max(len(df), 1) * 100, 2)
    
    return result


def unique_values_check(df: pd.DataFrame, config: dict) -> dict:
    """Unique value counts for each column."""
    if not config.get('unique_values', {}).get('enabled', True):
        return {}
    
    return {'unique_values_count': df.nunique(dropna=False).to_dict()}


def constant_values_check(df: pd.DataFrame, config: dict) -> dict:
    """Find columns with constant values."""
    if not config.get('constant_values', {}).get('enabled', True):
        return {}
    
    constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    return {'columns_with_constant_values': constant_columns}


def high_cardinality_check(df: pd.DataFrame, config: dict) -> dict:
    """Find categorical columns with high cardinality."""
    if not config.get('high_cardinality', {}).get('enabled', True):
        return {}
    
    threshold = config.get('high_cardinality', {}).get('threshold', 0.5)
    high_cardinality_cols = [
        col for col in df.select_dtypes(include=['object', 'category']).columns
        if df[col].nunique() > threshold * len(df)
    ]
    return {
        'columns_with_high_cardinality': high_cardinality_cols,
        'cardinality_threshold_used': threshold
    }


def primary_key_detection_check(df: pd.DataFrame, config: dict) -> dict:
    """Detect potential primary key columns."""
    if not config.get('primary_key_detection', {}).get('enabled', True):
        return {}
    
    potential_keys = [col for col in df.columns if df[col].is_unique and df[col].notna().all()]
    return {'potential_primary_keys': potential_keys}


def mixed_types_check(df: pd.DataFrame, config: dict) -> dict:
    """Find columns with mixed data types."""
    if not config.get('mixed_types', {}).get('enabled', True):
        return {}
    
    mixed_type_cols = []
    for col in df.select_dtypes(include='object').columns:
        # Sample approach: check if column contains both numeric and non-numeric strings
        sample_data = df[col].dropna().astype(str)
        if len(sample_data) > 0:
            numeric_count = pd.to_numeric(sample_data, errors='coerce').notna().sum()
            if 0 < numeric_count < len(sample_data):
                mixed_type_cols.append(col)
    
    return {'columns_with_mixed_types': mixed_type_cols}


def whitespace_issues_check(df: pd.DataFrame, config: dict) -> dict:
    """Find columns with whitespace issues."""
    if not config.get('whitespace_issues', {}).get('enabled', True):
        return {}
    
    whitespace_cols = []
    for col in df.select_dtypes(include='object').columns:
        non_null_series = df[col].dropna()
        if len(non_null_series) > 0:
            if non_null_series.apply(lambda x: str(x) != str(x).strip()).any():
                whitespace_cols.append(col)
    
    return {'columns_with_whitespace_issues': whitespace_cols}


def outlier_detection_check(df: pd.DataFrame, config: dict) -> dict:
    """Detect outliers in numeric columns."""
    if not config.get('outlier_detection', {}).get('enabled', True):
        return {}
    
    outlier_config = config.get('outlier_detection', {})
    method = outlier_config.get('method', 'iqr')
    outliers_count = {}
    
    for col in df.select_dtypes(include=np.number).columns:
        if method == 'iqr':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            multiplier = outlier_config.get('iqr_multiplier', 1.5)
            outlier_condition = (df[col] < (q1 - multiplier * iqr)) | (df[col] > (q3 + multiplier * iqr))
        elif method == 'zscore':
            threshold = outlier_config.get('zscore_threshold', 3)
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_condition = z_scores > threshold
        else:
            continue
            
        outliers_count[col] = int(outlier_condition.sum())
    
    return {
        'outliers_count': outliers_count,
        'outlier_method_used': method
    }


def value_distribution_check(df: pd.DataFrame, config: dict) -> dict:
    """Get value distribution for categorical columns."""
    if not config.get('value_distribution', {}).get('enabled', True):
        return {}
    
    top_n = config.get('value_distribution', {}).get('top_n', 5)
    value_distribution = {}
    
    for col in df.select_dtypes(include=['object', 'category']).columns:
        value_distribution[col] = df[col].value_counts(dropna=False).head(top_n).to_dict()
    
    return {
        'value_distribution_top_n': value_distribution,
        'top_n_used': top_n
    }


def date_analysis_check(df: pd.DataFrame, config: dict) -> dict:
    """Analyze datetime columns."""
    if not config.get('date_analysis', {}).get('enabled', True):
        return {}
    
    date_config = config.get('date_analysis', {})
    check_future = date_config.get('check_future_dates', True)
    date_info = {}
    
    for col in df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns:
        col_series = df[col].dropna()
        if not col_series.empty:
            info = {
                "min": col_series.min().isoformat(),
                "max": col_series.max().isoformat(),
                "range_days": (col_series.max() - col_series.min()).days
            }
            
            if check_future:
                info["future_dates_count"] = int((col_series > pd.Timestamp.utcnow()).sum())
            
            date_info[col] = info
    
    return {'date_column_analysis': date_info}


def legacy_custom_checks(df: pd.DataFrame, config: dict) -> dict:
    """
    Run the original custom checks (numeric_ranges, date_validation, value_sets).
    """
    results = {}
    checks = config.get('custom_checks', {})

    # Numeric range checks
    for rule in checks.get('numeric_ranges', []):
        col = rule['column']
        if col not in df.columns:
            results[f"numeric_range_{col}"] = 'column not found'
            continue
        series = pd.to_numeric(df[col], errors='coerce')
        low = rule.get('min', None)
        high = rule.get('max', None)
        mask = pd.Series([True]*len(series))
        if low is not None:
            mask &= series >= low
        if high is not None:
            mask &= series <= high
        invalid = (~mask & series.notnull()).sum()
        results[f"numeric_range_{col}"] = {
            'total_invalid': int(invalid),
            'percent_invalid': round(invalid/len(df)*100, 2)
        }

    # Date validation checks
    for rule in checks.get('date_validation', []):
        col = rule['column']
        allow_future = rule.get('allow_future_dates', False)
        if col not in df.columns:
            results[f"date_validation_{col}"] = 'column not found'
            continue
        parsed = pd.to_datetime(df[col], errors='coerce')
        invalid_dates = parsed.isna().sum()
        future_dates = 0
        if not allow_future:
            future_dates = (parsed > pd.Timestamp.utcnow()).sum()
        results[f"date_validation_{col}"] = {
            'invalid_dates': int(invalid_dates),
            'future_dates': int(future_dates)
        }

    # Value set checks
    for rule in checks.get('value_sets', []):
        col = rule['column']
        allowed = set(rule.get('allowed_values', []))
        if col not in df.columns:
            results[f"value_set_{col}"] = 'column not found'
            continue
        unique_vals = set(df[col].dropna().unique())
        disallowed = unique_vals - allowed
        results[f"value_set_{col}"] = {
            'disallowed_values': list(disallowed),
            'count_disallowed': int(df[col].isin(disallowed).sum())
        }

    return results


def generate_report(df: pd.DataFrame, config: dict) -> dict:
    """Generate comprehensive data quality report based on configuration."""
    # Merge with default config to ensure all options are available
    default_config = get_default_config()
    
    # Deep merge configs
    def deep_merge(default, custom):
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                deep_merge(default[key], value)
            else:
                default[key] = value
        return default
    
    merged_config = deep_merge(default_config.copy(), config)
    
    # Run all checks
    report = {}
    
    # Basic checks
    report.update(basic_stats_check(df, merged_config))
    report.update(null_analysis_check(df, merged_config))
    report.update(duplicate_analysis_check(df, merged_config))
    report.update(unique_values_check(df, merged_config))
    report.update(constant_values_check(df, merged_config))
    report.update(high_cardinality_check(df, merged_config))
    report.update(primary_key_detection_check(df, merged_config))
    report.update(mixed_types_check(df, merged_config))
    report.update(whitespace_issues_check(df, merged_config))
    report.update(outlier_detection_check(df, merged_config))
    report.update(value_distribution_check(df, merged_config))
    report.update(date_analysis_check(df, merged_config))
    
    # Legacy custom checks
    legacy_results = legacy_custom_checks(df, merged_config)
    if legacy_results:
        report['legacy_custom_checks'] = legacy_results
    
    # Add metadata
    report['generated_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
    
    return report

def visualize_report(report: dict, output_name: str = "dq_report.json", output_dir: str = "visualizations"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(output_name))[0]

    # Thresholds for meaningful visualization
    THRESHOLDS = {
        'null_percentage': 1.0,      # Show if > 1%
        'duplicate_rows_percent': 0.1,  # Show if > 0.1%
        'outliers_count': 5,         # Show if > 5 outliers
        'high_cardinality_threshold': 0.5  # Default from report
    }

    plots_needed = []
    clean_metrics = []
    text_summaries = []

    # 1. NULL Values Analysis
    if 'null_percentage' in report:
        nulls = pd.Series(report['null_percentage'])
        significant_nulls = nulls[nulls > THRESHOLDS['null_percentage']].sort_values(ascending=False)
        
        if len(significant_nulls) > 0:
            plots_needed.append(('nulls', significant_nulls))
        else:
            clean_metrics.append("✔ No significant null values (all < 1%)")

    # 2. Duplicate Rows Analysis
    dup_pct = report.get('duplicate_rows_percent', 0)
    if dup_pct > THRESHOLDS['duplicate_rows_percent']:
        plots_needed.append(('duplicates', dup_pct))
    else:
        clean_metrics.append("✔ No significant duplicate rows")

    # 3. Outliers Analysis
    if 'outliers_count' in report:
        outliers = pd.Series(report['outliers_count'])
        significant_outliers = outliers[outliers > THRESHOLDS['outliers_count']].sort_values(ascending=False)
        
        if len(significant_outliers) > 0:
            plots_needed.append(('outliers', significant_outliers))
        else:
            clean_metrics.append("✔ No significant outliers detected")

    # 4. Text-based summaries (no graphs needed)
    
    # High Cardinality Columns
    high_card_cols = report.get('columns_with_high_cardinality', [])
    if high_card_cols:
        unique_counts = {col: report['unique_values_count'][col] for col in high_card_cols if col in report.get('unique_values_count', {})}
        text_summaries.append(('High Cardinality Columns', unique_counts))
    else:
        clean_metrics.append("✔ No high cardinality columns")

    # Constant Value Columns  
    const_cols = report.get('columns_with_constant_values', [])
    if const_cols:
        text_summaries.append(('Constant Value Columns', {col: 1 for col in const_cols}))
    else:
        clean_metrics.append("✔ No constant value columns")

    # Potential Primary Keys
    pk_cols = report.get('potential_primary_keys', [])
    if pk_cols:
        unique_counts = {col: report['unique_values_count'][col] for col in pk_cols if col in report.get('unique_values_count', {})}
        text_summaries.append(('Potential Primary Keys', unique_counts))
    else:
        clean_metrics.append("✔ No strong primary key candidates")

    # Whitespace Issues
    ws_cols = report.get('columns_with_whitespace_issues', [])
    if ws_cols:
        text_summaries.append(('Whitespace Issues', {col: 'detected' for col in ws_cols}))
    else:
        clean_metrics.append("✔ No whitespace issues")

    # Mixed Type Columns
    mixed_cols = report.get('columns_with_mixed_types', [])
    if mixed_cols:
        text_summaries.append(('Mixed Type Columns', {col: 'detected' for col in mixed_cols}))
    else:
        clean_metrics.append("✔ No mixed type issues")

    # Calculate grid size
    total_plots = len(plots_needed) + len(text_summaries) + (1 if clean_metrics else 0)
    
    if total_plots == 0:
        print("✅ File passes all data quality checks. No issues found. Ready for analysis.")
        return

    # Create dynamic subplot grid
    if total_plots <= 2:
        nrows, ncols = 1, total_plots
    elif total_plots <= 4:
        nrows, ncols = 2, 2
    elif total_plots <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    if total_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten() if nrows * ncols > 1 else [axs]
    
    plot_idx = 0

    # Generate plots
    for plot_type, data in plots_needed:
        if plot_type == 'nulls':
            sns.barplot(x=data.values, y=data.index, palette="Reds_r", ax=axs[plot_idx])
            axs[plot_idx].set_title("Null Values by Column (%)")
            axs[plot_idx].set_xlabel("Null Percentage")
            
        elif plot_type == 'duplicates':
            axs[plot_idx].bar(['Duplicates', 'Clean Rows'], [data, 100 - data], 
                            color=['#ff4444', '#44ff44'])
            axs[plot_idx].set_title(f"Duplicate Rows ({data:.2f}%)")
            axs[plot_idx].set_ylabel("Percentage")
            
        elif plot_type == 'outliers':
            sns.barplot(x=data.values, y=data.index, palette="coolwarm", ax=axs[plot_idx])
            axs[plot_idx].set_title("Outliers by Column")
            axs[plot_idx].set_xlabel("Outlier Count")
        
        plot_idx += 1

    # Generate text summaries
    for title, data in text_summaries:
        if isinstance(data, dict):
            if title == 'Whitespace Issues' or title == 'Mixed Type Columns':
                text = "\n".join([f"• {col}" for col in data.keys()])
            else:
                text = "\n".join([f"• {col}: {count}" for col, count in data.items()])
        else:
            text = str(data)
            
        axs[plot_idx].text(0.05, 0.95, text, fontsize=10, verticalalignment='top', 
                          transform=axs[plot_idx].transAxes, family='monospace')
        axs[plot_idx].set_title(title, fontweight='bold')
        axs[plot_idx].axis('off')
        plot_idx += 1

    # Clean metrics summary
    if clean_metrics:
        summary_text = "\n".join(clean_metrics)
        axs[plot_idx].text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
                          transform=axs[plot_idx].transAxes, color='green', family='monospace')
        axs[plot_idx].set_title("Clean Checks Summary", fontweight='bold', color='green')
        axs[plot_idx].axis('off')
        plot_idx += 1

    # Turn off remaining axes
    for ax in axs[plot_idx:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Data quality report saved: {output_dir}/{base_name}_visualization.png")


# Bonus: Value Distribution Visualization Function
def visualize_value_distributions(report: dict, output_name: str = "report.json", output_dir: str = "visualizations"):
    """
    Visualize top value distributions for categorical columns.
    Useful for ETL pipelines to identify:
    - Skewed distributions requiring stratified sampling
    - Dominant categories for encoding strategies  
    - Rare values that could be grouped as 'Other'
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    if 'value_distribution_top_n' not in report:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(output_name))[0]
    
    distributions = report['value_distribution_top_n']
    
    # Filter out high-cardinality columns (like Name/ID fields)
    high_card_cols = set(report.get('columns_with_high_cardinality', []))
    filtered_dist = {col: dist for col, dist in distributions.items() 
                    if col not in high_card_cols and len(dist) > 1}
    
    if not filtered_dist:
        print("No meaningful value distributions to visualize")
        return
    
    n_cols = len(filtered_dist)
    fig, axs = plt.subplots(nrows=(n_cols+1)//2, ncols=2, figsize=(12, 4*((n_cols+1)//2)))
    if n_cols == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    
    for idx, (col, dist) in enumerate(filtered_dist.items()):
        # Calculate percentages
        total = sum(dist.values())
        percentages = {k: (v/total)*100 for k, v in dist.items()}
        
        # Create horizontal bar plot
        y_pos = range(len(percentages))
        values = list(percentages.values())
        labels = list(percentages.keys())
        
        bars = axs[idx].barh(y_pos, values, color=plt.cm.Set3(range(len(values))))
        axs[idx].set_yticks(y_pos)
        axs[idx].set_yticklabels(labels)
        axs[idx].set_xlabel('Percentage (%)')
        axs[idx].set_title(f'{col} - Top Values Distribution')
        
        # Add percentage labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axs[idx].text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                         f'{width:.1f}%', ha='left', va='center')
    
    # Turn off remaining axes
    for ax in axs[n_cols:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_value_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Value distribution report saved: {output_dir}/{base_name}_value_distributions.png")