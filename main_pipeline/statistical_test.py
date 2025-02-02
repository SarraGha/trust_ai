import json
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Load the JSON files
with open('evaluation_results_ADELE.json', 'r') as file:
    expert1_data = json.load(file)

with open('evaluation_results_SARRA.json', 'r') as file:
    expert2_data = json.load(file)


# Function to count responses by type and level
def count_responses(data):
    levels = ['a0', 'a1', 'a2', 'a3', 'a4']
    response_counts = {level: {'AI': 0, 'Human': 0, 'Both': 0} for level in levels}

    for key, responses in data.items():
        for response in responses:
            level = f"a{response['level']}"  # Convert level to a0, a1, etc.
            response_type = response['response_type']
            if response_type == 'AI':
                response_counts[level]['AI'] += 1
            elif response_type == 'Human':
                response_counts[level]['Human'] += 1
            elif response_type == 'Both':
                response_counts[level]['Both'] += 1

    return response_counts


# Count responses for Expert 1 and Expert 2 datasets
expert1_counts = count_responses(expert1_data)
expert2_counts = count_responses(expert2_data)

# Combine counts from both datasets
combined_counts = {level: {'AI': 0, 'Human': 0, 'Both': 0} for level in ['a0', 'a1', 'a2', 'a3', 'a4']}
for level in combined_counts:
    combined_counts[level]['AI'] = expert1_counts[level]['AI'] + expert2_counts[level]['AI']
    combined_counts[level]['Human'] = expert1_counts[level]['Human'] + expert2_counts[level]['Human']
    combined_counts[level]['Both'] = expert1_counts[level]['Both'] + expert2_counts[level]['Both']

# Aggregate counts for AI, Human, and Both responses
ai_count = np.sum([combined_counts[level]['AI'] for level in ['a0', 'a1', 'a2', 'a3', 'a4']])
human_count = np.sum([combined_counts[level]['Human'] for level in ['a0', 'a1', 'a2', 'a3', 'a4']])
both_count = np.sum([combined_counts[level]['Both'] for level in ['a0', 'a1', 'a2', 'a3', 'a4']])

# Total number of responses
total_responses = ai_count + human_count + both_count

# Perform a one-tailed proportion test
# Null hypothesis: AI + Both proportion <= Human proportion
# Alternative hypothesis: AI + Both proportion > Human proportion
count = np.array([ai_count + both_count, human_count])
nobs = np.array([total_responses, total_responses])
z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')

# Output results
print("=== Response Counts ===")
print(f"AI responses: {ai_count}")
print(f"Human responses: {human_count}")
print(f"Both are good: {both_count}")
print(f"Total responses: {total_responses}")
print("\n=== Statistical Test Results ===")
print(f"Z-statistic: {z_stat}")
print(f"P-value: {p_value}")