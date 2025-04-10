import matplotlib.pyplot as plt

# Model names
models = ['XGBClassifier','RandomForestClassifier','KNN', 'Naïve Bayes', 'SVM', 'Stacking Ensemble']

# Accuracy (in percentage)
accuracy = [74.11,74.08, 69.17, 64.94, 72.45, 73.63]

# Runtime (in seconds)
runtime = [40.15, 386.29, 6.17, 0.07, 272.4, 566.2]

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart for accuracy
bar = ax1.bar(models, accuracy, color='skyblue', label='Accuracy (%)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 100)
ax1.set_title('Model Accuracy and Runtime Comparison')

# Rotate x-axis labels
plt.xticks(rotation=45)

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
