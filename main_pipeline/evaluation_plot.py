import json
import matplotlib.pyplot as plt
import argparse

def load_json_file(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def count_responses(data):
    """Count the occurrences of each response type."""
    response_counts = {
        "AI": 0,
        "Human": 0,
        "Both": 0
    }

    for key, responses in data.items():
        for response in responses:
            response_type = response["response_type"]
            if response_type in response_counts:
                response_counts[response_type] += 1

    return response_counts

def plot_responses(file1_counts, file2_counts):
    """Plot the response counts for each file as a bar chart."""
    labels = list(file1_counts.keys())
    sizes1 = list(file1_counts.values())
    sizes2 = list(file2_counts.values())

    x = range(len(labels))  # the label locations

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35  # width of the bars

    # Create bars for each file with shades of blue
    bars1 = ax.bar(x, sizes1, bar_width, label='Expert 1', color='#1f77b4')  # Dark blue
    bars2 = ax.bar([p + bar_width for p in x], sizes2, bar_width, label='Expert 2', color='#66c2a5')  # Light blue

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Response Type', fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.set_title('Response Counts by Expert', fontsize=16)
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)

    # Annotate bars with counts
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def main(file1, file2):
    # Load data from the two JSON files
    data1 = load_json_file(file1)
    data2 = load_json_file(file2)

    # Count the responses for each file
    file1_counts = count_responses(data1)
    file2_counts = count_responses(data2)

    # Plot the results
    plot_responses(file1_counts, file2_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process two JSON files and plot response types.')
    parser.add_argument('file1', type=str, help='Path to the first JSON file')
    parser.add_argument('file2', type=str, help='Path to the second JSON file')

    args = parser.parse_args()
    main(args.file1, args.file2)