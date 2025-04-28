import json

import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Mapping levels to numeric noise
level_to_noise = {'A0': 0, 'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4}


# Read JSONL
def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


# Processing factual correctness file
def process_factual_correctness(df):
    df = df.copy()
    df['noise_level'] = df['level'].map(level_to_noise)
    df['method'] = df['type']
    df['llm'] = df['fact_extraction_llm']
    df['score'] = df['result']
    return df[['id', 'method', 'llm', 'noise_level', 'score']]


# Processing LLM-as-a-judge file
def process_llm_as_judge(df):
    df = df.copy()
    df['method'] = 'LLM-as-judge'
    df['llm'] = df['llm']
    df['score'] = df['result']
    # Assume noise_level must be provided separately!
    return df[['id', 'method', 'llm', 'score']]


# Merge factual and llm-as-judge datasets
def merge_data(factual_df, judge_df, noise_map):
    # Add noise levels into judge_df
    judge_df = judge_df.copy()
    judge_df['noise_level'] = judge_df['id'].map(noise_map)
    return pd.concat([factual_df, judge_df], ignore_index=True)


# Compute correlations
def compute_correlations(df):
    results = []

    for (method, llm), group in df.groupby(['method', 'llm']):
        noise_all = group['noise_level'].tolist()
        score_all = group['score'].tolist()

        if len(noise_all) > 0:
            pearson_corr, _ = pearsonr(noise_all, score_all)
            spearman_corr, _ = spearmanr(noise_all, score_all)
        else:
            pearson_corr = None
            spearman_corr = None

        results.append({
            'Method': method,
            'LLM': llm,
            'Pearson Correlation': pearson_corr,
            'Spearman Correlation': spearman_corr
        })

    return pd.DataFrame(results)


# Main
if __name__ == "__main__":
    factual_path = "factual_correctness.jsonl"
    judge_path = "llm_as_judge.jsonl"

    factual_df = load_jsonl(factual_path)
    judge_df = load_jsonl(judge_path)

    processed_factual = process_factual_correctness(factual_df)
    processed_judge = process_llm_as_judge(judge_df)

    # Create a noise map (id -> noise level) from factual correctness file
    noise_info = factual_df[['id', 'level']].drop_duplicates()
    noise_info['noise_level'] = noise_info['level'].map(level_to_noise)
    noise_map = dict(zip(noise_info['id'], noise_info['noise_level']))

    merged_df = merge_data(processed_factual, processed_judge, noise_map)
    results_table = compute_correlations(merged_df)

    print(results_table)
    results_table.to_csv('correlation_results.csv', index=False)
