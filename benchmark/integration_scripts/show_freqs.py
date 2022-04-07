import numpy as np
import matplotlib.pyplot as plt


with open("col_freqs.txt") as f:
    data = f.read()
data = data.split('\n')
y_cols = data

with open("row_freqs.txt") as f:
    data = f.read()
data = data.split('\n')
y_rows = data

x = range(0, max(len(y_cols), len(y_rows)))

y_cols += [0] * (max(len(y_cols), len(y_rows)) - len(y_cols))
y_rows += [0] * (max(len(y_cols), len(y_rows)) - len(y_rows))

print(len(x))
print(len(y_cols))
print(len(y_rows))

# plotting the line 1 points
plt.plot(x, y_cols, label = "col freqs")
plt.plot(x, y_rows, label = "row freqs")
plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
plt.title('Frequencies of graph vertex accesses')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.savefig("freqs.png")