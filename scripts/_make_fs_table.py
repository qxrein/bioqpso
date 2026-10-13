import json, numpy as np, os

with open('results/e8_feature_selection/e8_results.json') as f:
    d = json.load(f)

ALGO_DISPLAY = {
    'qpso':'QPSO', 'pso':'PSO', 'aco_r':'ACO$_R$', 'abc':'ABC',
    'gwo':'GWO',   'woa':'WOA', 'ant-fixed':'AntBio', 'bee-fixed':'BeeBio',
    'ant-cosine':'Ant-Cos', 'ant-feedback':'Ant-FB',
}
DS_SHORT = {
    'breast_cancer':'BC(30)', 'wine':'Wine(13)',
    'ionosphere':'Iono(34)', 'madelon_synth':'Mad(500)',
}

algos   = list(ALGO_DISPLAY.keys())
dsets   = list(DS_SHORT.keys())
n_ds    = len(dsets)

col_spec = 'l' + 'rr' * n_ds
header1  = 'Algorithm'
for ds in dsets:
    header1 += ' & \\multicolumn{2}{c}{' + DS_SHORT[ds] + '}'
header2 = ''
for _ in dsets:
    header2 += ' & kNN & \\#f'

lines = [
    '\\begin{tabular}{' + col_spec + '}',
    '\\toprule',
    header1 + ' \\\\',
    header2 + ' \\\\',
    '\\midrule',
]

for a in algos:
    row = ALGO_DISPLAY[a]
    for ds in dsets:
        if ds not in d:
            row += ' & --- & ---'
            continue
        res  = d[ds]['algorithms'][a]
        ph   = res['wilcoxon_p_holm']
        knn  = res['knn_mean']
        nf   = res['n_sel_mean']
        v_all = [d[ds]['algorithms'][aa]['knn_mean'] for aa in algos]
        best = abs(knn - max(v_all)) < 0.0015
        knn_s = f'{knn:.3f}'
        if ph < 0.05:  knn_s += '$^{*}$'
        if best:       knn_s = '\\textbf{' + knn_s + '}'
        row += f' & {knn_s} & {nf:.0f}'
    lines.append(row + ' \\\\')

lines.append('\\midrule')
for bl, bl_name in [('all_features','All-feat'), ('mi_topk','MI-k')]:
    row = bl_name
    for ds in dsets:
        if ds not in d:
            row += ' & --- & ---'
            continue
        res = d[ds]['baselines'][bl]
        row += f' & {res[\"knn_mean\"]:.3f} & {res[\"n_sel_mean\"]:.0f}'
    lines.append(row + ' \\\\')

lines += ['\\bottomrule', '\\end{tabular}']

out = 'paper/generated/fs.tex'
with open(out, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Wrote', out)
