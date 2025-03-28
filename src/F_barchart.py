import matplotlib.pyplot as plt

# Model names
models = ['KNN', 'Naïve Bayes', 'SVM']

# Accuracy (in percentage)
accuracy = [75.8, 68.1, 80.2]

# Runtime (in seconds)
runtime = [10.41, 0.05, 232.27]

# Create figure and axis
fig, ax1 = plt.subplots()

# Bar chart for accuracy
bar = ax1.bar(models, accuracy, color='skyblue', label='Accuracy (%)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 100)
ax1.set_title('Model Accuracy and Runtime Comparison')

# Add accuracy labels on top of bars
for i, v in enumerate(accuracy):
    ax1.text(i, v + 1, f"{v}%", ha='center')

# Create second y-axis for runtime
ax2 = ax1.twinx()
line = ax2.plot(models, runtime, color='orange', marker='o', label='Runtime (sec)')
ax2.set_ylabel('Runtime (sec)')
ax2.set_ylim(0, max(runtime) * 1.1)

# Combine legends
lines_labels = [bar, line[0]]
labels = [l.get_label() for l in lines_labels]
ax1.legend(lines_labels, labels, loc='upper left')

plt.tight_layout()
plt.show()