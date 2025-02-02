import matplotlib.pyplot as plt
import pandas as pd

# Load the evaluation results CSV file
file_path = "./ragas_evaluation_results.csv"
df = pd.read_csv(file_path)

# Assign answer levels based on row order (assuming repeating pattern A0, A1, A2, A3, A4 every 5 lines)
answer_levels = ["A0", "A1", "A2", "A3", "A4"]
df["answer_level"] = [answer_levels[i % 5] for i in range(len(df))]

# Group by answer level and compute the average correctness score for each level
level_scores = df.groupby("answer_level")["answer_correctness"].mean()

# Ensure the levels are in the correct order
level_scores = level_scores.reindex(["A0", "A1", "A2", "A3", "A4"])

# Drop NaN values in case some levels are missing
to_plot = level_scores.dropna()

# Plot the results
plt.figure(figsize=(8, 5))
plt.bar(to_plot.index, to_plot.values, color='skyblue')
plt.xlabel("Answer Level")
plt.ylabel("Average Answer Correctness Score")
plt.title("Overall Answer Correctness Score by Level")
plt.ylim(0, 1)  # Ensure the scale is from 0 to 1
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()
