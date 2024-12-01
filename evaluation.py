import matplotlib.pyplot as plt

def evaluate_performance(initial_times, optimized_times, function_names):
    x = list(range(len(function_names)))
    width = 0.35

    # Ensure all times are not None
    initial_times = [t if t is not None else 0 for t in initial_times]
    optimized_times = [t if t is not None else 0 for t in optimized_times]

    fig, ax = plt.subplots()
    rects1 = ax.bar([i - width/2 for i in x], initial_times, width, label='Initial')
    rects2 = ax.bar([i + width/2 for i in x], optimized_times, width, label='Optimized')

    ax.set_xlabel('Functions')
    ax.set_ylabel('Execution Time')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(function_names, rotation=90)
    ax.legend()

    # Use logarithmic scale for y-axis
    ax.set_yscale('log')

    # Add value labels to bars for better visibility
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2e}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
