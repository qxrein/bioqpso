#!/usr/bin/env python3
"""
E8 — Feature selection with nested CV (resumable, one JSON per algorithm and fold).

Protocol:
- Outer: stratified 5-fold x 6 repeats = 30 outer evaluations per (algorithm, dataset)
- Inner fitness: 5-fold k-NN accuracy on the outer-train split only
- Features standardised with statistics of the outer-train split
- Outer-test accuracy: (a) k-NN (k=5) and (b) RBF-SVM
- Fitness: 0.7*(1-acc_inner) + 0.3*|S|/D
- Datasets (real): Breast Cancer, Wine, Ionosphere, Madelon (OpenML)
- Baselines: all features, mutual-information top-D//4
- Budget: n=30; T=50 (1,530 FEs), T=20 (630 FEs) for Madelon
"""
from _runlib import *
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from bioqpso.boundary import BoundaryHandler
from bioqpso.budget import FEBudget
from bioqpso.runner import _make_algorithm

OUT = "results/e8_feature_selection/jobs"; os.makedirs(OUT, exist_ok=True)
N_PARTICLES, N_OUTER, N_REPEATS, N_INNER = 30, 5, 6, 5
DATASETS = {"breast_cancer": 50, "wine": 50, "ionosphere": 50, "madelon": 20}


def load(name):
    if name == "breast_cancer":
        d = load_breast_cancer(); return d.data.astype(float), d.target
    if name == "wine":
        d = load_wine(); return d.data.astype(float), d.target
    d = fetch_openml(name, version=1, as_frame=False, parser="liac-arff")
    return np.asarray(d.data, dtype=float), LabelEncoder().fit_transform(d.target)


def folds(X, y):
    out = []
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=SEEDS[rep] % (2**32))
        out += list(skf.split(X, y))
    return out


def fold_data(ds, fi):
    X, y = load(ds)
    tr, te = folds(X, y)[fi]
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X[tr]), y[tr], sc.transform(X[te]), y[te]


def test_scores(Xtr, ytr, Xte, yte, mask):
    Xa, Xb = Xtr[:, mask], Xte[:, mask]
    return (float(KNeighborsClassifier(5).fit(Xa, ytr).score(Xb, yte)),
            float(SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xa, ytr).score(Xb, yte)))


def job(path, ds, algo, fi):
    Xtr, ytr, Xte, yte = fold_data(ds, fi)
    D, T = Xtr.shape[1], DATASETS[ds]
    if algo == "all_features":
        mask = np.ones(D, bool)
    elif algo == "mi_topk":
        mi = mutual_info_classif(Xtr, ytr, random_state=42)
        mask = np.zeros(D, bool); mask[np.argsort(mi)[-max(1, D // 4):]] = True
    else:
        def fitness(x):
            m = np.asarray(x) >= 0.5
            if not m.any():
                return 1.0
            acc = cross_val_score(KNeighborsClassifier(5), Xtr[:, m], ytr,
                                  cv=N_INNER, scoring="accuracy").mean()
            return 0.7 * (1.0 - acc) + 0.3 * m.sum() / D
        seed = SEEDS[fi % len(SEEDS)]
        rng = np.random.default_rng(seed)
        bud = FEBudget(fitness, N_PARTICLES * (T + 1), soft=True)
        opt = _make_algorithm(algo, bud, 0.0, 1.0, D, N_PARTICLES, T, rng,
                              BoundaryHandler(0.0, 1.0, rng=rng), hist_every=T,
                              extra=FULL12[algo])
        opt.run()
        mask = opt.gbest_pos >= 0.5
    knn, svm = test_scores(Xtr, ytr, Xte, yte, mask) if mask.any() else (0.0, 0.0)
    write_json(path, dict(dataset=ds, algorithm=algo, fold=fi, final_f=0.0, knn=knn, svm=svm,
                          n_sel=int(mask.sum()), mask=np.flatnonzero(mask).tolist()))


if __name__ == "__main__":
    for ds in DATASETS:
        load(ds)  # fill the OpenML cache before forking
        run_jobs(f"E8-{ds}", job, [(f"{OUT}/{ds}__{a}__f{fi:02d}.json", (ds, a, fi))
                                   for a in list(FULL12) + ["all_features", "mi_topk"]
                                   for fi in range(N_OUTER * N_REPEATS)])
