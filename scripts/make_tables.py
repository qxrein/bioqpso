#!/usr/bin/env python3
"""
make_tables.py — the single source of every table and prose number in the paper.

Reads raw per-run files in results/, writes results/analysis/<exp>/*.csv,
paper/generated/*.tex and paper/generated/numbers.tex (macros used in the prose).
Experiments whose results are missing are skipped with a message.
"""
from __future__ import annotations
import csv, glob, json, os, re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats

OUT = "paper/generated"
ANA = "results/analysis"
os.makedirs(OUT, exist_ok=True)

ALGO = {
    "pso": ("PSO", "PSO"), "qpso": ("QPSO", "QPSO"), "aco_r": (r"ACO$_\mathbb{R}$", "ACOR"),
    "abc": ("ABC", "ABC"), "gwo": ("GWO", "GWO"), "woa": ("WOA", "WOA"),
    "lshade": ("L-SHADE", "LSHADE"), "ant-fixed": ("AntBio", "AntBio"),
    "bee-fixed": ("BeeBio", "BeeBio"), "ant-linear": ("Ant-Lin", "AntLin"),
    "ant-cosine": ("Ant-Cos", "AntCos"), "ant-feedback": ("Ant-FB", "AntFB"),
    "bee-per_particle": ("Bee-PP", "BeePP"),
}
ALGOS = list(ALGO)
BIO = ["ant-fixed", "bee-fixed", "ant-linear", "ant-cosine", "ant-feedback", "bee-per_particle"]
FN = {
    "sphere": ("Sphere", "Sphere"), "rosenbrock": ("Rosenbrock", "Rosenbrock"),
    "zakharov": ("Zakharov", "Zakharov"), "dixon_price": ("Dixon--Price", "DixonPrice"),
    "rastrigin": ("Rastrigin", "Rastrigin"), "ackley": ("Ackley", "Ackley"),
    "griewank": ("Griewank", "Griewank"), "schwefel": ("Schwefel", "Schwefel"),
    "levy": ("Levy", "Levy"), "styblinski_tang": ("Styblinski--Tang", "StyblinskiTang"),
}
FNS = list(FN)
CEC = [f"F{i}2017" for i in range(1, 30) if i != 5]
ST_OPT = -39.16616570377142 * 30  # f* of Styblinski-Tang at D=30 (see benchmarks/classical.py)
OPT = {"styblinski_tang": ST_OPT}
_macros: list[str] = []


def cec_label(f):
    return f.replace("2017", "")


# ── formatting ────────────────────────────────────────────────────────────
def sig(x, n=3):
    """Unified number format: n significant figures, scientific if |x|<1e-2 or >=1e4."""
    if x is None or not np.isfinite(x):
        return "---"
    if x == 0:
        return "0"
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a < 1e-2 or a >= 1e4:
        m, e = f"{a:.{n-1}e}".split("e")
        return f"{sign}{m}e{int(e)}"
    r = float(f"{a:.{n-1}e}")  # rounded to n significant figures
    if r >= 1e4:
        m, e = f"{a:.{n-1}e}".split("e")
        return f"{sign}{m}e{int(e)}"
    d = int(np.floor(np.log10(r)))
    return sign + f"{r:.{max(n-1-d, 0)}f}"


def tex(s, bold=False):
    """LaTeX math for a string from sig(). e notation is written x10^{}."""
    if s == "---":
        return s
    neg = s.startswith("-")
    body = s[1:] if neg else s
    if "e" in body:
        m, e = body.split("e")
        body = rf"{m}\mathrm{{e}}{{{e}}}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    return "$" + ("-" if neg else "") + body + "$"


def num(x, n=3):
    return tex(sig(x, n))


def pval(p):
    if not np.isfinite(p):
        return "---"
    if p >= 0.01:
        return f"{p:.3f}" if p < 0.9995 else "1"
    m, e = f"{p:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def mac(name, val):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    _macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def macn(name, x, n=3):
    mac(name, r"\ensuremath{" + tex(sig(x, n)).strip("$") + "}")


def macp(name, p):
    mac(name, r"\ensuremath{" + pval(p) + "}")


def letters(i):
    """Integer to letters, since macro names cannot contain digits."""
    return {30: "Thirty", 50: "Fifty", 100: "Hundred"}.get(i) or "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]


def fnmac(f):
    return FN[f][1] if f in FN else "F" + "".join(
        "Zero One Two Three Four Five Six Seven Eight Nine".split()[int(c)] for c in cec_label(f)[1:])


def fnlist(items, display=None):
    display = display or (lambda f: FN[f][0] if f in FN else cec_label(f))
    items = [display(i) for i in items]
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def write(name, lines):
    with open(f"{OUT}/{name}", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote", name)


def table(name, spec, header, rows, midrules=()):
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        lines.append(" & ".join(r) + r" \\")
        if i in midrules:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(name, lines)


# ── loading ───────────────────────────────────────────────────────────────
def load(pattern, key=lambda r: (r["algorithm"], r["function"]), field="final_f"):
    """{key: {run: value}} from all per-run files matching the glob."""
    out = defaultdict(dict)
    for p in glob.glob(pattern):
        with open(p) as f:
            r = json.load(f)
        out[key(r)][r["run"]] = float(r[field])
    return out


def vec(d, k):
    v = d.get(k)
    if not v:
        return None
    return np.array([v[i] for i in sorted(v)])


# ── statistics ────────────────────────────────────────────────────────────
def holm(ps):
    ps = np.asarray(ps, dtype=float)
    adj = np.ones_like(ps)
    running = 0.0
    for rank, idx in enumerate(np.argsort(ps)):
        running = max(running, ps[idx] * (len(ps) - rank))
        adj[idx] = min(running, 1.0)
    return adj


def wilcox(a, b, alternative="less"):
    """Paired Wilcoxon; identical samples give p=1."""
    if np.array_equal(a, b):
        return 1.0
    return float(stats.wilcoxon(a, b, alternative=alternative).pvalue)


def rel_impr(base, alt, fstar=0.0):
    """Relative improvement of alt over base on f - f*; positive if alt is better."""
    b, a = np.mean(base) - fstar, np.mean(alt) - fstar
    if b == 0:
        return 0.0 if a == 0 else -np.inf
    return float((b - a) / abs(b))


def rank_biserial(a, b):
    """Matched-pairs rank-biserial r; positive means a is lower (better) than b."""
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    rk = stats.rankdata(np.abs(d))
    return float((rk[d < 0].sum() - rk[d > 0].sum()) / rk.sum())


def rb_ci(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(a), size=(n_boot, len(a)))
    d = a[idx] - b[idx]
    rk = stats.rankdata(np.abs(d), axis=1)
    rk = np.where(d == 0, 0.0, rk)
    tot = rk.sum(axis=1)
    r = np.where(tot > 0, (rk * (d < 0)).sum(axis=1) - (rk * (d > 0)).sum(axis=1), 0.0) / np.where(tot > 0, tot, 1)
    return tuple(np.percentile(r, [2.5, 97.5]))


Q05 = {k: stats.studentized_range.isf(0.05, k, 1e6) / np.sqrt(2) for k in range(2, 20)}


def ranks_matrix(means):
    """means: (N functions, k algorithms); average ranks for ties (lower is better)."""
    return np.apply_along_axis(stats.rankdata, 1, means)


def friedman(means):
    k, N = means.shape[1], means.shape[0]
    R = ranks_matrix(means)
    chi2, p = stats.friedmanchisquare(*means.T)
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
    return R.mean(axis=0), float(chi2), float(p), float(cd), k, N


def means_for(d, algos, fns):
    return np.array([[np.mean(vec(d, (a, f))) for a in algos] for f in fns])


def complete(d, algos, fns, n=30):
    return all(len(d.get((a, f), {})) == n for a in algos for f in fns)


# ── value tables ──────────────────────────────────────────────────────────
def value_table(name, d, algos, fns, stat=np.mean, labels=None, first="Function"):
    """Per-function table in the unified format; best (lowest) per row is bold,
    including ties at the printed precision."""
    labels = labels or {a: ALGO[a][0] for a in algos}
    rows = []
    for f in fns:
        vals = [stat(vec(d, (a, f))) if vec(d, (a, f)) is not None else np.nan for a in algos]
        strs = [sig(v) for v in vals]
        fin = [(v, s) for v, s in zip(vals, strs) if np.isfinite(v)]
        best = min(fin)[1] if fin else None
        lab = FN[f][0] if f in FN else cec_label(f)
        rows.append([lab] + [tex(s, bold=(s == best)) for s in strs])
    table(name, "l" + "r" * len(algos), [first] + [labels[a] for a in algos], rows)


def winners(d, algos, fns):
    """{function: [algorithms with the lowest mean at printed precision]}."""
    w = {}
    for f in fns:
        vals = {a: np.mean(vec(d, (a, f))) for a in algos}
        best = sig(min(vals.values()))
        w[f] = [a for a, v in vals.items() if sig(v) == best]
    return w


def ranks_section(tag, d, algos, fns, control="qpso"):
    """Macros and a rank table for one experiment. Also writes the analysis CSV."""
    m = means_for(d, algos, fns)
    R, chi2, p, cd, k, N = friedman(m)
    mac(tag + "K", str(k)); mac(tag + "N", str(N))
    macn(tag + "Chi", chi2, 4); macp(tag + "P", p); macn(tag + "CD", cd)
    mac(tag + "Q", f"{Q05[k]:.3f}")
    order = np.argsort(R)
    for a, r in zip(algos, R):
        mac(tag + "Rank" + ALGO[a][1], f"{r:.2f}")
    mac(tag + "BestName", ALGO[algos[order[0]]][0])
    mac(tag + "SecondName", ALGO[algos[order[1]]][0])
    mac(tag + "ThirdName", ALGO[algos[order[2]]][0])
    mac(tag + "WorstName", ALGO[algos[order[-1]]][0])
    # Pairwise over function means against the control (Demsar's recommended alternative)
    ci = algos.index(control)
    others = [i for i in range(k) if i != ci]
    raw = [wilcox(m[:, i], m[:, ci], "two-sided") for i in others]
    adj = holm(raw)
    pw = {algos[i]: float(q) for i, q in zip(others, adj)}
    # Friedman ranks separated from the control by the CD
    sep = [algos[i] for i in others if abs(R[i] - R[ci]) > cd]
    mac(tag + "SepList", fnlist(sep, lambda a: ALGO[a][0]))
    sepb = [a for a in sep if R[algos.index(a)] < R[ci]]
    mac(tag + "SepBetterList", fnlist(sepb, lambda a: ALGO[a][0]))
    mac(tag + "SepBetterCount", str(len(sepb)))
    bio_better = [a for a in BIO if a in algos and R[algos.index(a)] < R[ci]]
    mac(tag + "BioBetterCount", str(len(bio_better)))
    mac(tag + "BioBetterList", fnlist(bio_better, lambda a: ALGO[a][0]))
    bio_sep = [a for a in BIO if a in sep]
    mac(tag + "BioSepCount", str(len(bio_sep)))
    mac(tag + "BioSepList", fnlist(bio_sep, lambda a: ALGO[a][0]))
    pw_sig = [a for a, q in pw.items() if q < 0.05 and a in BIO]
    mac(tag + "BioPwCount", str(len(pw_sig)))
    mac(tag + "BioPwList", fnlist(pw_sig, lambda a: ALGO[a][0]))
    for a, q in pw.items():
        macp(tag + "Pw" + ALGO[a][1], q)
    bio_r = [R[algos.index(a)] for a in BIO if a in algos]
    mac(tag + "BioBestRank", f"{min(bio_r):.2f}"); mac(tag + "BioWorstRank", f"{max(bio_r):.2f}")
    # win counts
    w = winners(d, algos, fns)
    for a in algos:
        won = [f for f in fns if a in w[f]]
        mac(tag + "Wins" + ALGO[a][1], str(len(won)))
        mac(tag + "WinFns" + ALGO[a][1], fnlist(won))
    rows = []
    for i in order:
        a = algos[i]
        rows.append([ALGO[a][0], f"{R[i]:.2f}", "---" if a == control else "$" + pval(pw[a]) + "$"])
    os.makedirs(f"{ANA}/{tag}", exist_ok=True)
    with open(f"{ANA}/{tag}/ranks.csv", "w", newline="") as fh:
        c = csv.writer(fh); c.writerow(["algorithm", "mean_rank", "holm_vs_control"])
        for i in order:
            c.writerow([algos[i], R[i], pw.get(algos[i], "")])
    json.dump(dict(chi2=chi2, p=p, cd=cd, k=k, N=N), open(f"{ANA}/{tag}/summary.json", "w"), indent=1)
    return rows, dict(R=dict(zip(algos, R)), cd=cd, chi2=chi2, p=p, pw=pw, k=k, N=N)


def ranks_tex(name, cols):
    """cols: list of (title, rows). Side-by-side rank tables."""
    n = max(len(r) for _, r in cols)
    spec = "".join("lrr" for _ in cols)
    hdr1 = " & ".join(rf"\multicolumn{{3}}{{c}}{{{t}}}" for t, _ in cols)
    hdr2 = " & ".join("Algorithm & Rank & $p_{\\mathrm{Holm}}$" for _ in cols)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", hdr1 + r" \\", hdr2 + r" \\", r"\midrule"]
    for i in range(n):
        lines.append(" & ".join(" & ".join(r[i]) if i < len(r) else "& &" for _, r in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(name, lines)


def wilcoxon_tables(d, algos, fns, fname_wtl, fname_eff, tag):
    """Win/tie/loss against QPSO with both directions tested, plus effect sizes."""
    others = [a for a in algos if a != "qpso"]
    ctrl = {f: vec(d, ("qpso", f)) for f in fns}
    res = {}
    for f in fns:
        less = holm([wilcox(vec(d, (a, f)), ctrl[f], "less") for a in others])
        grt = holm([wilcox(vec(d, (a, f)), ctrl[f], "greater") for a in others])
        for a, pl, pg in zip(others, less, grt):
            ri = rel_impr(ctrl[f], vec(d, (a, f)), OPT.get(f, 0.0))
            res[a, f] = "W" if pl < 0.05 and ri > 0.01 else "L" if pg < 0.05 and ri < -0.01 else "T"
    sym = {"W": r"\win", "T": r"\tie", "L": r"\loss"}
    rows, tot = [], {}
    for a in others:
        c = [res[a, f] for f in fns]
        tot[a] = (c.count("W"), c.count("T"), c.count("L"))
        rows.append([ALGO[a][0]] + [sym[x] for x in c] + [str(v) for v in tot[a]])
        mac(tag + "W" + ALGO[a][1], str(tot[a][0])); mac(tag + "T" + ALGO[a][1], str(tot[a][1]))
        mac(tag + "L" + ALGO[a][1], str(tot[a][2]))
    table(fname_wtl, "l" + "c" * len(fns) + "rrr",
          ["Algorithm"] + [(FN[f][0][:4] + ".") for f in fns] + ["W", "T", "L"], rows)
    # effect sizes for the BioQPSO family against QPSO
    rows = []
    for f in fns:
        cells = []
        for a in [x for x in BIO if x in algos]:
            r = rank_biserial(vec(d, (a, f)), ctrl[f])
            lo, hi = rb_ci(vec(d, (a, f)), ctrl[f])
            cells.append(f"${r:+.2f}$ {{\\tiny[{lo:+.2f},{hi:+.2f}]}}")
        rows.append([FN[f][0]] + cells)
    table(fname_eff, "l" + "r" * len([x for x in BIO if x in algos]),
          ["Function"] + [ALGO[a][0] for a in BIO if a in algos], rows)


# ── controls: pre-registered credit rules ─────────────────────────────────
def credit(alt, base, fns, opt=None):
    """Pre-registered test per function: one-sided Wilcoxon (Holm over functions), >1% rel."""
    opt = opt or {}
    raw = [wilcox(alt[f], base[f], "less") for f in fns]
    rawg = [wilcox(alt[f], base[f], "greater") for f in fns]
    adj, adjg = holm(raw), holm(rawg)
    out = {}
    for f, p, pg in zip(fns, adj, adjg):
        ri = rel_impr(base[f], alt[f], opt.get(f, 0.0))
        out[f] = dict(alt=float(np.mean(alt[f])), base=float(np.mean(base[f])), p=float(p), pg=float(pg),
                      rel=ri, credited=bool(p < 0.05 and ri > 0.01), harmed=bool(pg < 0.05 and ri < -0.01))
    return out


def bio_table(name, res, fns, leaders):
    rows = []
    for f in fns:
        row = [FN[f][0] if f in FN else cec_label(f)]
        for L in leaders:
            c = res[L][f]
            row += [num(c["alt"]), num(c["base"]), "$" + pval(c["p"]) + "$",
                    (f"{100*c['rel']:+.1f}\\%" if abs(c["rel"]) < 10 else (r"$>\!+10^3$\%" if c["rel"] > 0 else r"$<\!-10^3$\%")),
                    "Y" if c["credited"] else "N"]
        rows.append(row)
    hdr1 = " & " + " & ".join(rf"\multicolumn{{5}}{{c}}{{{L} leader}}" for L in leaders) + r" \\"
    hdr2 = "Function & " + " & ".join("Bio & No-bio & $p_{\\mathrm{Holm}}$ & Rel. & Cr." for _ in leaders) + r" \\"
    lines = [r"\begin{tabular}{l" + "rrrrc" * len(leaders) + "}", r"\toprule", hdr1, hdr2, r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(name, lines)


def controls(tag, d, fns, leaders, fixed, schedules, opt, tname):
    """Bio-leader and schedule rules; returns a results dict and emits macros."""
    bio = {}
    for L, (bio_key, nobio_key) in leaders.items():
        alt = {f: vec(d, (bio_key, f)) for f in fns}
        base = {f: vec(d, (nobio_key, f)) for f in fns}
        bio[L] = credit(alt, base, fns, opt)
        cr = [f for f in fns if bio[L][f]["credited"]]
        hm = [f for f in fns if bio[L][f]["harmed"]]
        mac(f"{tag}BioCount{L}", str(len(cr))); mac(f"{tag}BioFns{L}", fnlist(cr))
        mac(f"{tag}BioHarm{L}", str(len(hm))); mac(f"{tag}BioHarmFns{L}", fnlist(hm))
    bio_table(tname, bio, fns, list(leaders))
    # schedule vs best fixed (chosen per function on the mean, conservative)
    sched = {}
    for L, keys in fixed.items():
        best = {f: min(keys, key=lambda k: np.mean(vec(d, (k, f)))) for f in fns}
        base = {f: vec(d, (best[f], f)) for f in fns}
        for s, skey in schedules.get(L, {}).items():
            alt = {f: vec(d, (skey, f)) for f in fns}
            sched[s] = credit(alt, base, fns, opt)
            cr = [f for f in fns if sched[s][f]["credited"]]
            mac(f"{tag}SchedCount{s}", str(len(cr))); mac(f"{tag}SchedFns{s}", fnlist(cr))
            hm = [f for f in fns if sched[s][f]["harmed"]]
            mac(f"{tag}SchedHarm{s}", str(len(hm)))
    counts = {s: sum(c[f]["credited"] for f in fns) for s, c in sched.items()}
    mac(f"{tag}SchedMax", str(max(counts.values())) if counts else "0")
    mac(f"{tag}SchedTotal", str(sum(counts.values())))
    mac(f"{tag}SchedN", str(len(counts)))
    harmed = {s: sum(c[f]["harmed"] for f in fns) for s, c in sched.items()}
    mac(f"{tag}SchedHarmTotal", str(sum(harmed.values())))
    return bio, sched


def main():
    print("Loading results…")
    e1 = load("results/e1_classical/*.json")
    have_e1 = complete(e1, ALGOS, FNS)
    summary = {}
    rank_cols = []

    # ── E1 ──
    if have_e1:
        value_table("classical_mean.tex", e1, ALGOS, FNS)
        value_table("classical_std.tex", e1, ALGOS, FNS, stat=np.std)
        value_table("classical_median.tex", e1, ALGOS, FNS, stat=np.median)
        iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)
        value_table("classical_iqr.tex", e1, ALGOS, FNS, stat=iqr)
        rows, summary["E1"] = ranks_section("Eone", e1, ALGOS, FNS)
        rank_cols.append(("Classical (E1)", rows))
        wilcoxon_tables(e1, ALGOS, FNS, "wilcoxon.tex", "effect_sizes.tex", "Eone")
        for f in FNS:
            for a in ALGOS:
                v = vec(e1, (a, f))
                macn(f"EoneMean{FN[f][1]}{ALGO[a][1]}", np.mean(v))
                macn(f"EoneStd{FN[f][1]}{ALGO[a][1]}", np.std(v))
                macn(f"EoneMed{FN[f][1]}{ALGO[a][1]}", np.median(v))
                macn(f"EoneMax{FN[f][1]}{ALGO[a][1]}", np.max(v))
        # outliers: functions where mean and median of QPSO differ by more than 10x
        out = [f"{FN[f][0]}" for f in FNS
               if abs(np.mean(vec(e1, ("qpso", f))) - OPT.get(f, 0)) >
               10 * abs(np.median(vec(e1, ("qpso", f))) - OPT.get(f, 0))]
        mac("EoneQpsoOutlierFns", fnlist(out, str))
    else:
        print("  E1 incomplete; skipped")

    # ── E1b ──
    e1b = load("results/e1b_shifted/*.json")
    if complete(e1b, ALGOS, FNS):
        value_table("shifted_mean.tex", e1b, ALGOS, FNS)
        value_table("shifted_std.tex", e1b, ALGOS, FNS, stat=np.std)
        rows, summary["E1b"] = ranks_section("Eoneb", e1b, ALGOS, FNS)
        rank_cols.append(("Shifted (E1b)", rows))
        wilcoxon_tables(e1b, ALGOS, FNS, "wilcoxon_shifted.tex", "effect_sizes_shifted.tex", "Eoneb")
        for f in FNS:
            for a in ALGOS:
                macn(f"EonebMean{FN[f][1]}{ALGO[a][1]}", np.mean(vec(e1b, (a, f))))
        if have_e1:
            # origin-bias: factor by which GWO/WOA degrade under the shift on origin-optimum functions
            zero_opt = ["sphere", "zakharov", "rastrigin", "ackley", "griewank"]
            mac("OriginFns", fnlist(zero_opt))
            mac("GwoRankShift", f"{summary['E1']['R']['gwo']:.2f}")

    if rank_cols:
        ranks_tex("ranks.tex", rank_cols)

    # ── E2 ──
    e2 = load("results/e2_controls/*.json")
    keys2 = ["qpso", "ant-nobio", "bee-nobio"] + [f"{L}-fixed-p0{i}" for L in ("ant", "bee") for i in range(1, 6)]
    if complete(e2, keys2, FNS):
        leaders = {"Ant": ("ant-fixed-p03", "ant-nobio"), "Bee": ("bee-fixed-p03", "bee-nobio")}
        fixed = {L: [f"{L.lower()}-fixed-p0{i}" for i in range(1, 6)] for L in leaders}
        sc = {"Ant": {"AntLin": "ant-linear", "AntCos": "ant-cosine", "AntFB": "ant-feedback", "AntRand": "ant-random"},
              "Bee": {"BeeLin": "bee-linear", "BeeCos": "bee-cosine", "BeeFB": "bee-feedback", "BeePP": "bee-per_particle", "BeeRand": "bee-random"}}
        bio, sched = controls("Etwo", e2, FNS, leaders, fixed, sc, OPT, "bioleader_credit.tex")
        summary["E2"] = dict(bio=bio, sched=sched)
        # phi3=0 is exactly the no-bio control after the fix
        same = all(np.array_equal(vec(e2, (f"{L}-fixed-p00", f)), vec(e2, (f"{L}-nobio", f)))
                   for L in ("ant", "bee") for f in FNS if vec(e2, (f"{L}-fixed-p00", f)) is not None)
        print("  phi3=0 identical to no-bio:", same)
        for f in FNS:
            for k, m in [("qpso", "QPSO"), ("ant-nobio", "AntNobio"), ("bee-nobio", "BeeNobio"),
                         ("ant-fixed-p03", "AntFixed"), ("bee-fixed-p03", "BeeFixed")]:
                macn(f"EtwoMean{FN[f][1]}{m}", np.mean(vec(e2, (k, f))))
        # decomposition: removing the random phi (QPSO → no-bio) and adding the leader (no-bio → bio)
        for L, key in (("Ant", "ant-nobio"), ("Bee", "bee-nobio")):
            base = {f: vec(e2, ("qpso", f)) for f in FNS}
            alt = {f: vec(e2, (key, f)) for f in FNS}
            dec = credit(alt, base, FNS, OPT)
            better = [f for f in FNS if dec[f]["credited"]]
            worse = [f for f in FNS if dec[f]["harmed"]]
            mac(f"DecBetter{L}", str(len(better))); mac(f"DecBetterFns{L}", fnlist(better))
            mac(f"DecWorse{L}", str(len(worse))); mac(f"DecWorseFns{L}", fnlist(worse))
            summary[f"dec{L}"] = dec
        # fixed-phi3 table
        rows2 = [("QPSO", "qpso"), ("no-bio ($\\phi_3=0$)", "ant-nobio")] + \
                [(f"$\\phi_3=0.{i}$", f"ant-fixed-p0{i}") for i in range(1, 6)] + \
                [("random", "ant-random"), ("Linear", "ant-linear"), ("Cosine", "ant-cosine"), ("Feedback", "ant-feedback")]
        lab = {k: l for l, k in rows2}
        keys = [k for _, k in rows2]
        value_table("controls_fixed_sweep.tex", e2, keys, FNS, labels=lab)
        rows2b = [("QPSO", "qpso"), ("no-bio ($\\phi_3=0$)", "bee-nobio")] + \
                 [(f"$\\phi_3=0.{i}$", f"bee-fixed-p0{i}") for i in range(1, 6)] + \
                 [("random", "bee-random"), ("Linear", "bee-linear"), ("Cosine", "bee-cosine"),
                  ("Feedback", "bee-feedback"), ("PerPart.", "bee-per_particle")]
        value_table("controls_fixed_sweep_bee.tex", e2, [k for _, k in rows2b], FNS,
                    labels={k: l for l, k in rows2b})
        # Spread of the fixed sweep: how flat is the phi3 landscape?
        flat = []
        for f in FNS:
            base = np.array([np.mean(vec(e2, (f"ant-fixed-p0{i}", f))) - OPT.get(f, 0) for i in range(1, 6)])
            if base.max() <= 2 * max(base.min(), 1e-300):
                flat.append(f)
        mac("EtwoFlatCount", str(len(flat))); mac("EtwoFlatFns", fnlist(flat))
    else:
        print("  E2 incomplete; skipped")

    # ── E3 light ──
    e3 = load("results/e3_cec2017_d30/*.json")
    if complete(e3, ALGOS, CEC):
        half = len(CEC) // 2
        value_table("cec2017_mean_a.tex", e3, ALGOS, CEC[:half], first="Fn")
        value_table("cec2017_mean_b.tex", e3, ALGOS, CEC[half:], first="Fn")
        value_table("cec2017_std_a.tex", e3, ALGOS, CEC[:half], stat=np.std, first="Fn")
        value_table("cec2017_std_b.tex", e3, ALGOS, CEC[half:], stat=np.std, first="Fn")
        rows, summary["E3"] = ranks_section("Ethree", e3, ALGOS, CEC)
        cec_cols = [("Light, $D=30$ (E3)", rows)]
        R, q = summary["E3"]["R"], "qpso"
        gaps = {a: R[q] - R[a] for a in BIO}
        macn("EthreeMaxGap", max(gaps.values()))
        ant_r = [R[a] for a in ("ant-fixed", "ant-linear", "ant-cosine", "ant-feedback")]
        mac("EthreeAntBest", f"{min(ant_r):.2f}"); mac("EthreeAntWorst", f"{max(ant_r):.2f}")
    else:
        cec_cols = []
        print("  E3 incomplete; skipped")

    # ── E2c ──
    e2c = load("results/e2c_controls_cec/*.json")
    if complete(e3, ALGOS, CEC) and complete(e2c, ["ant-nobio", "bee-nobio", "ant-fixed-p01", "ant-fixed-p05", "ant-random"], CEC):
        c = dict(e2c)
        for src, dst in [("qpso", "qpso"), ("ant-fixed", "ant-fixed-p03"), ("bee-fixed", "bee-fixed-p03"),
                         ("ant-linear", "ant-linear"), ("ant-cosine", "ant-cosine"),
                         ("ant-feedback", "ant-feedback"), ("bee-per_particle", "bee-per_particle")]:
            for f in CEC:
                c[dst, f] = e3[src, f]
        leaders = {"Ant": ("ant-fixed-p03", "ant-nobio"), "Bee": ("bee-fixed-p03", "bee-nobio")}
        fixed = {"Ant": ["ant-fixed-p01", "ant-fixed-p03", "ant-fixed-p05"]}
        sc = {"Ant": {"AntLin": "ant-linear", "AntCos": "ant-cosine", "AntFB": "ant-feedback", "AntRand": "ant-random"}}
        bio, sched = controls("Etwoc", c, CEC, leaders, fixed, sc, {}, "bioleader_credit_cec.tex")
        summary["E2c"] = dict(bio=bio, sched=sched)
        for L, key in (("Ant", "ant-nobio"), ("Bee", "bee-nobio")):
            dec = credit({f: vec(c, (key, f)) for f in CEC}, {f: vec(c, ("qpso", f)) for f in CEC}, CEC)
            better = [f for f in CEC if dec[f]["credited"]]
            worse = [f for f in CEC if dec[f]["harmed"]]
            mac(f"DecCecBetter{L}", str(len(better))); mac(f"DecCecWorse{L}", str(len(worse)))
        keys = ["qpso", "ant-nobio", "ant-fixed-p01", "ant-fixed-p03", "ant-fixed-p05", "ant-random",
                "ant-cosine", "bee-nobio", "bee-fixed-p03"]
        lab = dict(zip(keys, ["QPSO", "no-bio", "$\\phi_3{=}0.1$", "$\\phi_3{=}0.3$", "$\\phi_3{=}0.5$",
                              "random", "Cosine", "Bee no-bio", "Bee $0.3$"]))
        value_table("controls_cec.tex", c, keys, CEC, labels=lab, first="Fn")
        # Do the controls change the rank picture? rank QPSO, no-bio, AntBio, Bee, Bee no-bio on CEC
        sub = ["qpso", "ant-nobio", "ant-fixed-p03", "bee-nobio", "bee-fixed-p03"]
        Rs = ranks_matrix(means_for(c, sub, CEC)).mean(axis=0)
        for k, r in zip(["QPSO", "AntNobio", "AntBio", "BeeNobio", "BeeBio"], Rs):
            mac("EtwocRank" + k, f"{r:.2f}")
    else:
        print("  E2c incomplete; skipped")

    # ── E3s ──
    for D, dn in ((30, "Thirty"), (50, "Fifty")):
        d = load(f"results/e3s_cec2017_d{D}/*.json")
        tag = "Ethrees" + dn
        have = {a for (a, f), v in d.items() if len(v) == 30}
        al = [a for a in ALGOS if all(len(d.get((a, f), {})) == 30 for f in CEC)]
        if len(al) >= 3 and "qpso" in al:
            half = len(CEC) // 2
            value_table(f"e3s_d{D}_mean_a.tex", d, al, CEC[:half], first="Fn")
            value_table(f"e3s_d{D}_mean_b.tex", d, al, CEC[half:], first="Fn")
            value_table(f"e3s_d{D}_std_a.tex", d, al, CEC[:half], stat=np.std, first="Fn")
            value_table(f"e3s_d{D}_std_b.tex", d, al, CEC[half:], stat=np.std, first="Fn")
            rows, summary[f"E3s{D}"] = ranks_section(tag, d, al, CEC)
            cec_cols.append((f"Standard, $D={D}$", rows))
            mac(tag + "AlgoCount", str(len(al)))
        else:
            print(f"  E3s D={D} incomplete; skipped")
    if cec_cols:
        ranks_tex("cec2017_ranks.tex", cec_cols)

    # ── E7 / E7b ──
    e7 = load("results/e7_schwefel/*.json", key=lambda r: (r["algorithm"], r["boundary_mode"]))
    al7 = ["qpso", "ant-nobio", "ant-fixed", "bee-fixed", "ant-cosine"]
    lab7 = {"qpso": "QPSO", "ant-nobio": "no-bio (ant)", "ant-fixed": "AntBio", "bee-fixed": "BeeBio", "ant-cosine": "Ant-Cos"}
    modes = ["clip", "reflect", "random_reinit", "absorb"]
    if all(len(e7.get((a, m), {})) == 30 for a in al7 for m in modes):
        rows = []
        for m, name in zip(modes[:3], ["clip", "reflect", "random re-init."]):
            rows.append([name] + [tex(sig(np.mean(vec(e7, (a, m))))) for a in al7])
        table("e7_schwefel.tex", "l" + "r" * len(al7), ["Boundary"] + [lab7[a] for a in al7], rows)
        for a in al7:
            for m, n in zip(modes[:3], ["Clip", "Reflect", "Reinit"]):
                macn("Esev" + lab7[a].replace(" ", "").replace("-", "").replace("(", "").replace(")", "") + n,
                     np.mean(vec(e7, (a, m))))
    e7b = load("results/e7b_schwefel_shift/*.json", key=lambda r: (r["algorithm"], r["optimum"], r["boundary"]))
    al7b = ALGOS + ["ant-nobio"]
    if all(len(e7b.get((a, t, b), {})) == 30 for a in al7b for t in ("near", "inward") for b in ("clip", "reflect")):
        rows = []
        for t, tl in (("near", "$c=420.97$"), ("inward", "$c=120.97$")):
            for b in ("clip", "reflect"):
                vals = [np.mean(vec(e7b, (a, t, b))) for a in al7b]
                strs = [sig(v) for v in vals]
                best = strs[int(np.argmin(vals))]
                rows.append([tl, b] + [tex(s, s == best) for s in strs])
        table("e7b_schwefel.tex", "ll" + "r" * len(al7b),
              ["Optimum", "Boundary"] + [ALGO[a][0] if a in ALGO else "no-bio" for a in al7b], rows)
        for t, T in (("near", "Near"), ("inward", "Inward")):
            for b, B in (("clip", "Clip"), ("reflect", "Reflect")):
                for a in al7b:
                    nm = ALGO[a][1] if a in ALGO else "Nobio"
                    macn(f"EsevB{T}{B}{nm}", np.mean(vec(e7b, (a, t, b))))
                # gap of the fixed bio variant and the no-bio control to QPSO
                q = vec(e7b, ("qpso", t, b))
                for a, nm in (("ant-fixed", "Ant"), ("bee-fixed", "Bee"), ("ant-nobio", "Nobio")):
                    v = vec(e7b, (a, t, b))
                    macn(f"EsevBRatio{T}{B}{nm}", np.mean(v) / np.mean(q))
                    macp(f"EsevBP{T}{B}{nm}", wilcox(v, q, "two-sided"))
        fam = ["ant-fixed", "bee-fixed", "ant-linear", "ant-cosine", "ant-feedback", "bee-per_particle"]
        better = {}
        for t in ("near", "inward"):
            better[t] = sum(wilcox(vec(e7b, (a, t, "clip")), vec(e7b, ("qpso", t, "clip")), "greater") < 0.05 / 6
                            for a in fam)
        mac("EsevBWorseNear", str(better["near"])); mac("EsevBWorseInward", str(better["inward"]))
        summary["E7b"] = {"near": better["near"], "inward": better["inward"]}

    # ── E4 ──
    e4 = load("results/e4_scalability/*.json", key=lambda r: (r["algorithm"], r["function"], r["dim"]))
    al4 = ["qpso", "gwo", "ant-fixed", "ant-cosine", "ant-feedback", "bee-fixed", "bee-per_particle"]
    fn4 = ["sphere", "rastrigin", "rosenbrock", "levy", "schwefel", "griewank"]
    n4 = sum(len(v) for v in e4.values())
    mac("EfourJobs", f"{n4:,}".replace(",", "{,}"))
    if all(len(e4.get((a, f, D), {})) == 30 for a in al4 for f in fn4 for D in (30, 50, 100)):
        rows, mids = [], []
        for f in fn4:
            for D in (30, 50, 100):
                m = [np.mean(vec(e4, (a, f, D))) for a in al4]
                s = [np.std(vec(e4, (a, f, D))) for a in al4]
                st = [sig(v) for v in m]
                best = st[int(np.argmin(m))]
                rows.append([FN[f][0] if D == 30 else "", str(D)] +
                            [tex(x, x == best) + " {\\tiny(" + tex(sig(y)) + ")}" for x, y in zip(st, s)])
            mids.append(len(rows) - 1)
        table("scalability.tex", "lr" + "r" * len(al4), ["Function", "$D$"] + [ALGO[a][0] for a in al4], rows, mids[:-1])
        for D in (30, 50, 100):
            m = np.array([[np.mean(vec(e4, (a, f, D))) for a in al4] for f in fn4])
            R = ranks_matrix(m).mean(axis=0)
            dn = letters(D)
            mac(f"EfourLeader{dn}", ALGO[al4[int(np.argmin(R))]][0])
            for a, r in zip(al4, R):
                mac(f"EfourRank{dn}{ALGO[a][1]}", f"{r:.2f}")
            wins = winners({(a, f): e4[a, f, D] for a in al4 for f in fn4}, al4, fn4)
            for a in al4:
                won = [f for f in fn4 if a in wins[f]]
                mac(f"EfourWins{dn}{ALGO[a][1]}", str(len(won)))
                mac(f"EfourWinFns{dn}{ALGO[a][1]}", fnlist(won))
        zeros = {f: all(np.all(vec(e4, ("gwo", f, D)) == 0) for D in (30, 50, 100)) for f in fn4}
        mac("EfourGwoZeroFns", fnlist([f for f in fn4 if zeros[f]]))
        mac("EfourGwoZeroCount", str(sum(zeros.values())))
        summary["E4"] = True

    # ── E6 ──
    e6 = load("results/e6_sensitivity/*.json", key=lambda r: (r["param"], r["value"], r["function"]))
    if e6:
        make_e6(e6, summary)

    # ── E8 ──
    make_e8(summary)

    # ── E9 ──
    if os.path.exists("results/e9_runtime/runtime_results.json"):
        make_e9()

    with open(f"{OUT}/numbers.tex", "w") as f:
        f.write("% Auto-generated by scripts/make_tables.py — do not edit\n")
        f.write("\n".join(_macros) + "\n")
    print(f"  wrote numbers.tex ({len(_macros)} macros)")
    os.makedirs(ANA, exist_ok=True)
    json.dump(summary, open(f"{ANA}/summary_all.json", "w"), indent=1, default=float)


SENS = {
    "rho": r"$\rho$ (AntBio)", "Delta": r"$\Delta$ (AntBio)", "Lscout": r"$L_{\mathrm{scout}}$ (BeeBio)",
    "W": r"$W$ (Ant-FB)", "delta": r"$\delta$ (Ant-FB)", "phi3bounds": r"$[\phi_3^{\min},\phi_3^{\max}]$ (Ant-Cos)",
    "beta": r"$\beta$ start--end (AntBio)",
}
SENS_FNS = ["sphere", "rastrigin", "levy", "styblinski_tang", "schwefel", "griewank", "F12017", "F42017", "F112017", "F212017"]


def make_e6(e6, summary):
    params = defaultdict(list)
    for (p, v, f) in e6:
        if v not in params[p]:
            params[p].append(v)
    if not all(len(e6.get((p, v, f), {})) == 30 for p in SENS for v in params[p] for f in SENS_FNS):
        print("  E6 incomplete; skipped")
        return
    rows, mids = [], []
    tests = []
    for p in SENS:
        vals = params[p]
        try:
            vals = sorted(vals, key=lambda s: tuple(float(x) for x in s.split("-")))
        except ValueError:
            pass
        params[p] = vals
        for v in vals:
            cells = [(sig(np.mean(vec(e6, (p, v, f))))) for f in SENS_FNS]
            rows.append([SENS[p] if v == vals[0] else "", v.replace("-", "--")] + [tex(c) for c in cells])
        mids.append(len(rows) - 1)
        for f in SENS_FNS:
            stat, q = stats.friedmanchisquare(*[vec(e6, (p, v, f)) for v in vals])
            tests.append((p, f, q))
    table("e6_sensitivity.tex", "ll" + "r" * len(SENS_FNS),
          ["Parameter", "Value"] + [(FN[f][0][:5] + "." if f in FN else cec_label(f)) for f in SENS_FNS],
          rows, mids[:-1])
    adj = holm([t[2] for t in tests])
    sig_cells = [(t[0], t[1]) for t, q in zip(tests, adj) if q < 0.05]
    mac("EsixCells", str(len(tests))); mac("EsixSigCells", str(len(sig_cells)))
    per = {p: [f for q, f in sig_cells if q == p] for p in SENS}
    for p, fs in per.items():
        name = {"rho": "Rho", "Delta": "BigDelta", "Lscout": "Lscout", "W": "W", "delta": "Delta",
                "phi3bounds": "Bounds", "beta": "Beta"}[p]
        mac(f"EsixSig{name}", str(len(fs))); mac(f"EsixSigFns{name}", fnlist(fs))
    summary["E6"] = {p: per[p] for p in SENS}
    json.dump({"params": params}, open(f"{ANA}/e6_params.json", "w"))


def make_e8(summary):
    files = glob.glob("results/e8_feature_selection/jobs/*.json")
    if not files:
        print("  E8 absent; skipped")
        return
    data = defaultdict(dict)
    for p in files:
        r = json.load(open(p))
        data[r["dataset"], r["algorithm"]][r["fold"]] = r
    ds_all = {"breast_cancer": "Breast Cancer", "wine": "Wine", "ionosphere": "Ionosphere", "madelon": "Madelon"}
    selectors = [a for a in ALGOS if a != "lshade"]
    if not all(len(data.get((d, a), {})) == 30 for d in ds_all for a in selectors + ["all_features", "mi_topk"]):
        print("  E8 incomplete; skipped")
        return
    get = lambda d, a, k: np.array([data[d, a][i][k] for i in range(30)], dtype=float)
    res = {}
    for clf in ("knn", "svm"):
        rows = []
        for a in selectors:
            rows.append(a)
        tbl = {}
        for d in ds_all:
            others = [a for a in selectors if a != "qpso"]
            raw = [wilcox(get(d, a, clf), get(d, "qpso", clf), "two-sided") for a in others]
            adj = dict(zip(others, holm(raw)))
            tbl[d] = adj
        res[clf] = tbl
    header1 = " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for n in ds_all.values()) + r" \\"
    header2 = "Algorithm & " + " & ".join("kNN & SVM & \\#f" for _ in ds_all) + r" \\"
    lines = [r"\begin{tabular}{l" + "rrr" * len(ds_all) + "}", r"\toprule", header1, header2, r"\midrule"]
    fmt = lambda v: f"{v:.3f}"
    best = {(d, c): max(fmt(np.mean(get(d, a, c))) for a in selectors) for d in ds_all for c in ("knn", "svm")}
    credited = {}
    for a in selectors + ["all_features", "mi_topk"]:
        cells = [ALGO[a][0] if a in ALGO else {"all_features": "All features", "mi_topk": "MI top-$k$"}[a]]
        for d in ds_all:
            for c in ("knn", "svm"):
                v = np.mean(get(d, a, c)); s = fmt(v)
                t = rf"\mathbf{{{s}}}" if a in selectors and s == best[d, c] else s
                mark = ""
                if a in res[c][d] and res[c][d][a] < 0.05:
                    mark = "^{*}"
                    if v - np.mean(get(d, "qpso", c)) > 0.005:
                        mark = "^{\\dagger}"
                        credited.setdefault((d, c), []).append(a)
                cells.append(f"${t}{mark}$")
            cells.append(f"{np.mean(get(d, a, 'n_sel')):.1f}")
        lines.append(" & ".join(cells) + r" \\")
        if a == selectors[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("fs.tex", lines)
    names = {"breast_cancer": "BC", "wine": "Wine", "ionosphere": "Iono", "madelon": "Mad"}
    disp = lambda a: ALGO[a][0]
    for d, n in names.items():
        for c, C in (("knn", "Knn"), ("svm", "Svm")):
            mac(f"Fs{n}{C}All", f"{np.mean(get(d, 'all_features', c)):.3f}")
            mac(f"Fs{n}{C}Mi", f"{np.mean(get(d, 'mi_topk', c)):.3f}")
            mac(f"Fs{n}{C}Qpso", f"{np.mean(get(d, 'qpso', c)):.3f}")
            mac(f"Fs{n}{C}Ant", f"{np.mean(get(d, 'ant-fixed', c)):.3f}")
            means = {a: np.mean(get(d, a, c)) for a in selectors}
            b = max(means, key=means.get)
            mac(f"Fs{n}{C}BestName", disp(b)); mac(f"Fs{n}{C}Best", f"{means[b]:.3f}")
            mac(f"Fs{n}{C}Credited", fnlist(credited.get((d, c), []), disp))
            mac(f"Fs{n}{C}CreditedCount", str(len(credited.get((d, c), []))))
            bio_cr = [a for a in credited.get((d, c), []) if a in BIO]
            mac(f"Fs{n}{C}BioCredited", str(len(bio_cr)))
            mac(f"Fs{n}{C}AllBeats", "yes" if np.mean(get(d, "all_features", c)) > max(means.values()) else "no")
            mac(f"Fs{n}{C}MiBeats", "yes" if np.mean(get(d, "mi_topk", c)) > max(means.values()) else "no")
        mac(f"Fs{n}FeatQpso", f"{np.mean(get(d, 'qpso', 'n_sel')):.1f}")
        mac(f"Fs{n}FeatAnt", f"{np.mean(get(d, 'ant-fixed', 'n_sel')):.1f}")
    total_cr = sum(len(v) for v in credited.values())
    bio_total = sum(len([a for a in v if a in BIO]) for v in credited.values())
    mac("FsCreditTotal", str(total_cr)); mac("FsBioCreditTotal", str(bio_total))
    # dominance summaries computed over both classifiers
    all_beats = [ds_all[d] for d in ds_all if all(
        np.mean(get(d, "all_features", c)) > max(np.mean(get(d, a, c)) for a in selectors) for c in ("knn", "svm"))]
    mi_beats = [ds_all[d] for d in ds_all if all(
        np.mean(get(d, "mi_topk", c)) > max(np.mean(get(d, a, c)) for a in selectors) for c in ("knn", "svm"))]
    mac("FsAllBeatsList", fnlist(all_beats, str)); mac("FsMiBeatsList", fnlist(mi_beats, str))
    mac("FsAllBeatsCount", str(len(all_beats))); mac("FsMiBeatsCount", str(len(mi_beats)))
    summary["E8"] = {f"{d}/{c}": v for (d, c), v in credited.items()}


def make_e9():
    rt = json.load(open("results/e9_runtime/runtime_results.json"))
    fns = list(rt)
    algos = [a for a in ALGOS if a in rt[fns[0]]]
    agg = {a: float(np.median([rt[f][a]["per_1k_fe_s"] for f in fns])) for a in algos}
    iqr = {a: float(np.median([rt[f][a]["iqr_s"] for f in fns])) for a in algos}
    med = {a: float(np.median([rt[f][a]["median_s"] for f in fns])) for a in algos}
    q = agg["qpso"]
    rows = []
    for a in algos:
        ov = 100 * (agg[a] / q - 1)
        rows.append([ALGO[a][0], f"{1000*agg[a]:.1f}", f"{med[a]:.2f}", f"{iqr[a]:.2f}",
                     "---" if a == "qpso" else f"${ov:+.0f}$\\%"])
        if a != "qpso":
            mac("RtOver" + ALGO[a][1], f"{abs(ov):.0f}\\%")
            mac("RtDir" + ALGO[a][1], "slower" if ov > 0 else "faster")
    table("runtime.tex", "lrrrr", ["Algorithm", "ms / 1000 FEs", "Median run (s)", "IQR (s)", "vs QPSO"], rows)


if __name__ == "__main__":
    main()
