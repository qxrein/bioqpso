import json, numpy as np, os

with open('results/e9_runtime/runtime_results.json') as f:
    rt = json.load(f)

ALGO_DISPLAY = {
    'pso': 'PSO', 'qpso': 'QPSO', 'aco_r': r'ACO$_R$', 'abc': 'ABC',
    'gwo': 'GWO', 'woa': 'WOA', 'ant-fixed': 'AntBio', 'bee-fixed': 'BeeBio',
    'ant-linear': 'Ant-Lin', 'ant-cosine': 'Ant-Cos',
    'ant-feedback': 'Ant-FB', 'bee-per_particle': 'Bee-PP',
}
ALGO_ORDER = list(ALGO_DISPLAY.keys())

algos = [a for a in ALGO_ORDER if a in next(iter(rt.values()))]
algo_med = {a: [] for a in algos}
algo_per1k = {a: [] for a in algos}
for fn, data in rt.items():
    for a in algos:
        if a in data:
            algo_med[a].append(data[a]['median_s'])
            algo_per1k[a].append(data[a]['per_1k_fe_s'])

qpso_med = np.mean(algo_med['qpso'])

lines = [
    r'\begin{tabular}{lrrr}',
    r'\toprule',
    r'Algorithm & Median s/run & ms/1k FE & Overhead \\',
    r'\midrule',
]
for a in algos:
    m  = np.mean(algo_med[a])
    p  = np.mean(algo_per1k[a]) * 1000
    oh = (m - qpso_med) / qpso_med * 100
    d  = ALGO_DISPLAY[a]
    sign = '+' if oh >= 0 else ''
    lines.append(f'{d} & {m:.3f} & {p:.1f} & {sign}{oh:.0f}\\% ' + '\\\\')
lines += [r'\bottomrule', r'\end{tabular}']

out = 'paper/generated/runtime.tex'
with open(out, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'Wrote {out}')

# Append runtime macros to numbers.tex
with open('paper/generated/numbers.tex', 'a') as f:
    f.write('\n% Runtime numbers\n')
    for a in ['ant-fixed', 'ant-feedback', 'bee-fixed', 'bee-per_particle', 'gwo']:
        oh = (np.mean(algo_med[a]) - qpso_med) / qpso_med * 100
        key = a.replace('-', '').replace('_', '').capitalize()
        sign = '+' if oh >= 0 else ''
        f.write(f'\\newcommand{{\\numOverhead{key}}}{{{sign}{oh:.0f}\\%}}\n')
print('Updated numbers.tex with runtime macros')
