import matplotlib.pyplot as plt

# List of queries
queries = [
    "Impact of social media on mental health",
    "WHO health advisories",
    "Weather forecast New York",
    "BBC news headlines",
    "Global warming statistics 2024",
    "Yahoo Finance stock market updates",
    "Shakespeare's influence on modern literature",
    "Quantum computing vs classical computing",
    "NASA Mars mission updates",
    "Key events in AI development 2024",
    "Market analysis of electric vehicles 2024",
    "Best noise-canceling headphones 2024",
    "Consumer Reports washing machines",
    "Python decorators",
    "Workplace safety measures during COVID-19",
    "CDC flu prevention",
    "Latest trends in renewable energy 2024",
    "TechCrunch technology news",
    "Evolution of jazz music",
    "MetMuseum latest exhibits",
]

# Number of successful results for each query
successful_results = [
    0, 5, 5, 5, 0, 5, 5, 5, 5, 1, 3, 5, 5, 5, 2, 4, 0, 5, 4, 5
]

# Calculate the percentage of successful results
total_tests = 5
success_rate = [(result / total_tests) * 100 for result in successful_results]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plotting the bar chart
bars = ax.barh(queries, success_rate, color="skyblue")

# Adding labels and title
ax.set_xlabel("Success Rate (%)")
ax.set_title("Success Rate for Each Query (Based on 5 Test Attempts)")

# Adding text annotations
for bar in bars:
    width = bar.get_width()
    label_x_pos = width - 10 if width > 10 else width + 1
    ax.text(
        label_x_pos,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        ha="center",
        va="center",
        color="black",
        fontsize=8,
    )

# Customize the tick labels for better readability
plt.yticks(fontsize=10)
plt.xticks(range(0, 101, 10))

# Adding figure note
description_text = (
    "Figure H2: Graphical representation of the success rates for each query tested. "
    "Each query was tested 5 times, and the success rate is calculated as the percentage of successful "
    "attempts out of these 5 tests. The chart compares the effectiveness of different queries, providing "
    "a clear visualization of the success rate for each query."
)
plt.figtext(
    0.5, -0.1, description_text, wrap=True, horizontalalignment="center", fontsize=10
)

# Display the chart
plt.tight_layout()
plt.show()


