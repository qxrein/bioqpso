import matplotlib.pyplot as plt
import numpy as np

functions = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank']
algorithms = ['PSO', 'QPSO', 'AntBioQPSO', 'BeeBioQPSO']

mean_values = np.array([
    [7.32e-14, 1.84e-11, 1.95e-07, 2.39e-01],
    [1.12e+02, 2.75e+01, 5.33e+01, 3.83e+01],  
    [1.65e+00, 1.07e-06, 8.23e-05, 2.52e-01],  
    [3.11e+00, 9.76e-03, 2.63e-02, 2.53e-01],  
]).T  
plt.style.use('default')
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
})

x = np.arange(len(functions))
bar_width = 0.18

colors = ['#E74C3C', '#2E86C1', '#239B56', '#AF7AC5']  

fig, ax = plt.subplots(figsize=(10, 5))

for i in range(len(algorithms)):
    ax.bar(x + i * bar_width - 1.5 * bar_width,
           mean_values[i],
           width=bar_width,
           label=algorithms[i],
           color=colors[i],
           edgecolor='black',
           linewidth=0.8)

ax.set_xlabel('Benchmark Functions', fontweight='bold')
ax.set_ylabel('Accuracy (Mean Value)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(functions, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.8)
ax.legend(frameon=True, edgecolor='black', loc='upper right')

plt.tight_layout()
plt.savefig('accuracy_bar_graph.png', dpi=400, bbox_inches='tight')
plt.show()
