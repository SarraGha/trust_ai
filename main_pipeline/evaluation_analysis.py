import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from bokeh.io import export_svg
from bokeh.models import Legend, ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import dodge
from scipy import stats
from statsmodels.stats.inter_rater import cohens_kappa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs=2, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    data = {"expert": [], "id": [], "noise_level": [], "preference": []}
    for exp, file in enumerate(args.input_files):
        with open(file, "r") as f:
            evaluation = json.load(f)

        for _id, question in sorted(evaluation.items(), key=lambda i: int(i[0])):
            for noise_level, assessment in sorted(question.items(), key=lambda i: i[0]):
                data["expert"].append(exp)
                data["id"].append(_id)
                data["noise_level"].append(noise_level)
                data["preference"].append(assessment)

    df = pd.DataFrame(data)

    # One-sided Z-test for H1: p_AI - p_Human > delta
    filtered = df
    df0 = filtered[filtered['expert'] == 0][['id', 'noise_level', 'preference']]
    df1 = filtered[filtered['expert'] == 1][['id', 'noise_level', 'preference']]

    df0 = df0.rename(columns={'preference': 'pref_0'})
    df1 = df1.rename(columns={'preference': 'pref_1'})

    merged = pd.merge(df0, df1, on=['id', 'noise_level'])
    contingency_table = pd.crosstab(merged['pref_0'], merged['pref_1'])

    print("Contingency table (excluding A0)")
    print(contingency_table)

    print()
    print("Bootstrap Confidence Interval for Non-Inferiority")


    def score_pair(pref0: str, pref1: str) -> float:
        """
        +1: if one prefer AI and the other prefer Both/Neither
        -1: if one prefer Expert and the other prefer Both/Neither
        +0: if no evaluators do not show any preference
        """
        ai = (pref0 == "AI") or (pref1 == "AI")
        expert = (pref0 == "Expert") or (pref1 == "Expert")
        return 1 if ai and not expert else (-1 if expert and not ai else 0)


    scores = np.array([score_pair(row[1]["pref_0"], row[1]["pref_1"]) for row in merged.iterrows()])
    data = (scores,)
    bootstrap_result = stats.bootstrap(data, np.mean, confidence_level=0.90, method="percentile", alternative="greater")
    print(f"mean={np.mean(scores)} ci={bootstrap_result.confidence_interval}")

    full_ties = merged[
        (merged["pref_0"].isin(["Both are good", "Neither are good"])) &
        (merged["pref_1"].isin(["Both are good", "Neither are good"]))
        ].shape[0]

    partial_ties = merged[
        ((merged["pref_0"].isin(["AI", "Expert"]) &
          merged["pref_1"].isin(["Both are good", "Neither are good"]))) |
        ((merged["pref_1"].isin(["AI", "Expert"]) &
          merged["pref_0"].isin(["Both are good", "Neither are good"])))
        ].shape[0]

    mixed_ties = merged[
        (merged["pref_0"].isin(["Both are good", "Neither are good"])) &
        (merged["pref_1"].isin(["Both are good", "Neither are good"])) &
        (merged["pref_0"] != merged["pref_1"])
        ].shape[0]

    total_ties = full_ties + partial_ties + mixed_ties
    print(f"Ties (A0 excluded): {total_ties / len(merged)}")

    # statistical power
    print()
    print("Statistical Power")

    observed_mean = np.mean(scores)
    observed_std = np.std(scores)
    # Parameters
    n_simulations = 1000  # Number of simulated datasets
    n_samples = 40  # Your sample size (N=40)
    delta = 0.1  # Non-inferiority margin
    alpha = 0.05  # Significance level
    power = []

    for _ in range(n_simulations):
        # Simulate data under the alternative hypothesis (true mean = observed_mean)
        simulated_scores = np.random.normal(loc=observed_mean, scale=observed_std, size=n_samples)

        # Run bootstrap test
        result = stats.bootstrap(
            (simulated_scores,),
            np.mean,
            confidence_level=0.90,
            alternative="greater"
        )

        # Check if lower CI bound exceeds -delta
        if result.confidence_interval.low > -delta:
            power.append(1)
        else:
            power.append(0)

    # Calculate empirical power
    power = np.mean(power)
    print(f"Estimated Power: {power:.2f}")

    # Cohen's kappa
    print()
    contingency_table_square = contingency_table.copy()
    contingency_table_square["Both are bad"] = 0
    contingency_table_square = contingency_table_square[
        ["AI", "Both are bad", "Both are good", "Expert"]
    ]
    kappa = cohens_kappa(contingency_table_square.to_numpy())
    print(f"Cohen’s Kappa: {kappa['kappa']:.3f}")

    # image - Preference by noise level
    grouped = df.groupby(['noise_level', 'preference']).size().unstack(fill_value=0)

    grouped = grouped.sort_index()
    grouped = grouped.reindex(["A0", "A1", "A2", "A3", "A4"])
    custom_pref_order = ["AI", "Expert", "Both are good", "Both are bad"]
    grouped = grouped[custom_pref_order]

    preferences = list(grouped.index)
    data = {'noise_level': preferences}
    for pref in custom_pref_order:
        data[pref] = grouped[pref].values

    source = ColumnDataSource(data)

    p = figure(x_range=preferences, y_range=(0, grouped.values.max() + 5),
               height=300, width=450, toolbar_location=None, tools="")
    p.add_layout(Legend(), 'right')

    colors = ["#03a343", "#d64e0f", "#0f8dd6", "#d90f23"]
    width = 0.18
    offsets = [2 * -0.18 + 0.09, -0.08, 0.10, 0.11 + 0.18]  # Adjust depending on number of preferences

    for i, (pref, label, color) in enumerate(
            zip(custom_pref_order, ["Prefers AI pipeline", "Prefers Expert", "Both are good", "Both are bad"], colors)
    ):
        p.vbar(x=dodge('noise_level', offsets[i], range=p.x_range),
               top=pref, width=width, source=source,
               color=color, legend_label=label)

    p.x_range.range_padding = 0.
    p.xgrid.grid_line_color = None
    p.legend.location = "top_right"
    p.legend.orientation = "vertical"
    p.xaxis.axis_label = "Noise Level"
    p.yaxis.axis_label = "Count"

    export_svg(p, filename=args.output_dir / "by_level.svg")

    # image - Counts by expert
    grouped = df.groupby(["preference", "expert"]).size().unstack(fill_value=0)
    grouped = grouped.reindex(["AI", "Expert", "Both are bad", "Both are good"])
    grouped = grouped.rename(index={"AI": "Prefers AI", "Expert": "Prefers Expert"})
    preferences = list(grouped.index)
    data = {'preference': preferences}
    for pref in range(2):
        data[f"{pref}"] = grouped[pref].values

    source = ColumnDataSource(data)

    p2 = figure(x_range=preferences, y_range=(0, grouped.values.max() + 5),
                height=300, width=350, toolbar_location=None, tools="")
    # p2.add_layout(Legend(), 'right')

    # Define colors for each preference
    colors = ["#030ea3", "#373d94"]
    width = 0.18
    offsets = [-0.1, 0.09]  # Adjust depending on number of preferences

    # Add bars
    for i, color in enumerate(colors):
        p2.vbar(x=dodge('preference', offsets[i], range=p2.x_range),
                top=f"{i}", width=width, source=source,
                color=color, legend_label=f"Expert {i + 1}")

    p2.x_range.range_padding = 0.
    p2.xgrid.grid_line_color = None
    p2.legend.location = "top_left"
    p2.legend.orientation = "vertical"
    p2.xaxis.axis_label = "Preference decision"
    p2.yaxis.axis_label = "Count"
    p2.xaxis.major_label_orientation = -np.pi / 8

    export_svg(p2, filename=args.output_dir / "by_expert_preference.svg")
