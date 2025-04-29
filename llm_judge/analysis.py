import argparse
import json
import pathlib
from itertools import cycle

import numpy as np
import pandas as pd
from bokeh.io import export_svgs
from bokeh.models import FactorRange, ColumnDataSource, Whisker, LabelSet
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.transform import jitter
from scipy.stats import pearsonr, kendalltau

# Mapping levels to numeric noise
level_to_noise = {'A0': 0, 'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4}


# Read JSONL
def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def process_factual_correctness(df):
    df = df.copy()
    df['noise_level'] = df['level'].map(level_to_noise)
    df['method'] = df['type']
    df['llm'] = df['fact_extraction_llm']
    df['score'] = df['result']
    return df[['id', 'noise_level', 'method', 'llm', 'score']]


def process_llm_as_judge(df):
    df = df.copy()
    df['noise_level'] = df['level'].map(level_to_noise)
    df['method'] = 'LLM-as-judge'
    df['llm'] = df['llm']
    df['score'] = df['result']
    return df[['id', 'noise_level', 'method', 'llm', 'score']]


def bootstrap_corr(x, y, corr_fn, n_bootstrap=10000):
    rng = np.random.default_rng()
    stats = []
    x = np.array(x)
    y = np.array(y)

    for _ in range(n_bootstrap):
        idxs = rng.integers(0, len(x), len(x))
        try:
            stat, _ = corr_fn(x[idxs], y[idxs])
            stats.append(stat)
        except:
            continue

    mean = np.mean(stats)
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return mean, lower, upper


# Compute correlations
def compute_correlations(df, n_bootstrap=10000):
    results = []
    rng = np.random.default_rng()

    for (method, llm), group in df.groupby(['method', 'llm']):
        # Pearson correlation (global)
        pearson_r, _ = pearsonr(group['noise_level'], group['score'])

        # Pearson bootstrap CI
        pearson_boot = []
        values = group[['noise_level', 'score']].values
        for _ in range(n_bootstrap):
            sample = values[rng.choice(len(values), size=len(values), replace=True)]
            r, _ = pearsonr(sample[:, 0], sample[:, 1])
            pearson_boot.append(r)
        pearson_low, pearson_high = np.percentile(pearson_boot, [2.5, 97.5])

        # Per-question Kendall's Tau
        per_question = (
            group.groupby('id')
            .apply(lambda g: kendalltau(g['noise_level'].values, g['score'].values).statistic, include_groups=False)
            .dropna()
        )
        mean_kendall_tau = per_question.mean()

        # Spearman bootstrap CI
        kendall_boot = []
        for _ in range(n_bootstrap):
            sample = rng.choice(per_question.values, size=len(per_question), replace=True)
            kendall_boot.append(np.mean(sample))
        kendall_low, kendall_high = np.percentile(kendall_boot, [2.5, 97.5])

        results.append({
            'Method': method,
            'LLM': llm,
            'Pearson Correlation': round(pearson_r, 3),
            'Pearson 95% CI': f"[{pearson_low:.3f}, {pearson_high:.3f}]",
            'Kendall (mean per question)': round(mean_kendall_tau, 3),
            'Kendall 95% CI': f"[{kendall_low:.3f}, {kendall_high:.3f}]"
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual-correctness", type=pathlib.Path, required=True)
    parser.add_argument("--llm-as-judge", type=pathlib.Path, required=True)
    args = parser.parse_args()

    factual_df = load_jsonl(args.factual_correctness)
    judge_df = load_jsonl(args.llm_as_judge)

    processed_factual = process_factual_correctness(factual_df)
    processed_judge = process_llm_as_judge(judge_df)

    processed_judge.loc[processed_judge["llm"] == "prometheus", "llm"] = "prometheus-v2:7b"
    processed_judge.loc[processed_judge["llm"] == "gemma3:12b-it-qat", "llm"] = "gemma3:12b"

    merged_df = pd.concat([processed_factual, processed_judge], ignore_index=True)
    results_table = compute_correlations(merged_df)

    print(results_table)
    results_table.to_csv('correlation_results.csv', index=False)

    # ---------- Strip Plot: Per-question Kendall's Tau ----------
    kendall_rows = []
    grouped = merged_df.groupby(['method', 'llm', 'id'])
    for (method, llm, qid), group in grouped:
        group_sorted = group.sort_values('noise_level')
        tau, _ = kendalltau(group_sorted['noise_level'], group_sorted['score'])
        kendall_rows.append({'method': method, 'llm': llm, 'id': qid, 'kendall_tau': tau})

    df_tau = pd.DataFrame(kendall_rows)

    # Define disambiguated x positions
    df_tau['x_key'] = df_tau['method'] + ' | ' + df_tau['llm']

    # Assign colors based on method
    method_colors = {
        "LLM-as-judge": Category10[10][0],
        "fast-fc": Category10[10][2],
        "default": Category10[10][2]
    }
    df_tau['color'] = df_tau['method'].map(method_colors)

    # Sort keys for plotting
    x_keys = list(df_tau['x_key'].unique())

    # Set up figure
    p1 = figure(
        x_range=FactorRange(*x_keys),
        height=250,
        width=400,
        output_backend="svg",
    )
    p1.yaxis.axis_label = "Kendall's Tau"

    # Plot points by method with legend
    for method in df_tau['method'].unique():
        method_df = df_tau[df_tau['method'] == method]
        source = ColumnDataSource(method_df)
        assert method is not None
        p1.scatter(
            x=jitter('x_key', width=0.25, range=p1.x_range),
            y='kendall_tau',
            source=source,
            size=3,
            alpha=0.6,
            color=method_colors[method],
            legend_label=method
        )

    p1.legend.title = "Method"
    p1.legend.location = "top_left"
    p1.xaxis.major_label_orientation = -np.pi / 8

    export_svgs(p1, filename="kendall_tau_stripplot.svg")


    # ---------- Line Plot: Score vs. Noise Level with 95% CI ----------
    def normalize_score(row):
        if row['method'] == 'LLM-as-judge':
            return (row['score'] - 1) / 4  # map [1,5] → [0,1]
        return row['score']


    results_table['model'] = results_table['Method'] + ' | ' + results_table['LLM']
    best_model = results_table.loc[results_table['Pearson Correlation'].idxmin(), 'model']  # most negative correlation
    worst_model = results_table.loc[results_table['Pearson Correlation'].idxmax(), 'model']  # least negative correlation
    models_to_plot = [best_model, worst_model]

    merged_df['score_normalized'] = merged_df.apply(normalize_score, axis=1)

    summary = (merged_df.groupby(['method', 'llm', 'noise_level'])['score_normalized']
               .agg(['mean', 'std', 'count'])
               .reset_index())
    summary['ci95'] = 1.96 * summary['std'] / np.sqrt(summary['count'])
    summary['model'] = summary['method'] + ' | ' + summary['llm']

    filtered_summary = summary[summary['model'].isin(models_to_plot)]

    p2 = figure(
        height=250,
        width=400,
        x_axis_label="Noise Level",
        y_axis_label="Average Score",
        output_backend="svg"
    )

    palette = cycle(['#1f77b4', '#d62728'])  # Just two colors

    for model_name in filtered_summary['model'].unique():
        sub = filtered_summary[filtered_summary['model'] == model_name]
        color = next(palette)
        p2.line(sub['noise_level'], sub['mean'], legend_label=model_name, line_width=2, color=color)
        p2.scatter(sub['noise_level'], sub['mean'], size=5, color=color)
        p2.segment(x0=sub['noise_level'], y0=sub['mean'] - sub['ci95'],
                   x1=sub['noise_level'], y1=sub['mean'] + sub['ci95'],
                   line_width=1, color=color)

    p2.legend.title = "Best/Worst Pearson Models"
    p2.legend.location = "top_right"
    p2.legend.label_text_font_size = "10pt"
    p2.legend.click_policy = "hide"
    # Export
    export_svgs(p2, filename="score_vs_noise.svg")


    # ---------- Size Plot: Person vs. LLM size ----------
    def parse_ci(ci):
        if isinstance(ci, str) and ci.startswith("["):
            try:
                lower, upper = ci.strip("[]").split(", ")
                return float(lower), float(upper)
            except:
                return np.nan, np.nan
        return np.nan, np.nan


    def extract_model_size(llm_name):
        import re
        match = re.search(r'(\d+(?:\.\d+)?)([bm])', llm_name.lower())
        if not match:
            return np.nan
        size = float(match.group(1))
        unit = match.group(2)
        return size * (1 if unit == 'b' else 1e-3)


    results_table["model_size"] = results_table["LLM"].apply(extract_model_size)

    results_table[["pearson_lower", "pearson_upper"]] = results_table["Pearson 95% CI"].apply(
        lambda x: pd.Series(parse_ci(x))
    )

    p3 = figure(
        x_axis_type="log",
        x_axis_label="Model Size (billion parameters)",
        y_axis_label="Pearson Correlation",
        height=250,
        width=400,
        output_backend="svg"
    )

    colors = Category10[10]
    methods = results_table["Method"].unique()

    for i, method in enumerate(methods):
        df = results_table[results_table["Method"] == method].dropna(subset=["model_size", "Pearson Correlation"])
        color = colors[i % len(colors)]

        source = ColumnDataSource(df)

        p3.scatter("model_size", "Pearson Correlation", source=source, size=10, color=color, legend_label=method)

        has_ci = df[["pearson_lower", "pearson_upper"]].notna().all(axis=1)
        if has_ci.any():
            err_source = ColumnDataSource(df[has_ci])
            p3.add_layout(Whisker(source=err_source, base="model_size",
                                  upper="pearson_upper", lower="pearson_lower", line_color=color))

    p3.legend.title = "Method"
    p3.legend.location = "bottom_right"
    p3.legend.click_policy = "hide"
    p3.y_range.start = -1.05
    p3.y_range.end = -0.45
    p3.xaxis.major_label_orientation = -np.pi / 4

    export_svgs(p3, filename="pearson_vs_llm_size.svg")
