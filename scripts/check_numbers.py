#!/usr/bin/env python3
"""Fail if raw digits remain in the prose of manuscript.tex; every number must be a macro
from paper/generated/numbers.tex. Tables, equations, labels, citations and the whitelist are
exempt. Usage: check_numbers.py [file]. Exit 1 and list the offending lines otherwise."""
import re, sys

WHITELIST = [
    r"D\s*=\s*30", r"\b30 runs\b", r"\b30 independent runs\b", r"(19|20)\d\d", r"Eq\.~?\(?\d*", r"\b[Ff]\d+\b",
    r"E\d+[a-z]?\b", r"\$\d+\$", r"\bQ[12]\b", r"\bk-?NN\b", r"\b10\{,\}000\s*D\b", r"10\^4", r"\b5-fold\b", r"\b1\\?%",
    r"\b30\{,\}030\b", r"\bn\s*=\s*30\b", r"T_\{?\\max\}?\s*=\s*1000", r"\[0\.1,\s*0\.5\]",
    # symbols, not quantities
    r"\\phi_\{?[123k]", r"\\chi\^2", r"\\phi_\{3,i\}", r"q_\{0\.05\}", r"\(6N\)", r"\$D=(30|50|100)\$",
    r"\\tfrac\d\d", r"\$4\{:\}3\$", r"\\phi_\d\^", r"\(1[-+]", r"[+-]1\b", r",1\)", r"U\(0,1\)", r"\\le0\.5",
    r"\\ge0", r"\\frac1", r"_\{i=1\}",
]
STRIP_ENVS = ["table", "sidewaystable", "figure", "equation", "algorithm", "algorithmic", "tabular",
              "thebibliography"]


def main(path="manuscript.tex"):
    s = open(path).read()
    body = s[s.index(r"\begin{document}"):]
    # Lines marked "% const" state protocol constants or dates, not results
    body = "\n".join("" if line.rstrip().endswith("% const") else line for line in body.split("\n"))
    body = re.sub(r"(?<!\\)%.*", "", body)
    for e in STRIP_ENVS:
        body = re.sub(rf"\\begin\{{{e}\*?\}}.*?\\end\{{{e}\*?\}}",
                      lambda m: "\n" * m.group(0).count("\n"), body, flags=re.S)
    body = re.sub(r"\\(ref|label|cite|eqref|input|geninput|includegraphics|url|email|affil\*?|todo|texttt|bibliography\w*)(\[[^\]]*\])?\{[^}]*\}", "", body)
    bad = []
    for n, line in enumerate(body.split("\n"), 1):
        t = line
        for w in WHITELIST:
            t = re.sub(w, "", t)
        t = re.sub(r"\\[A-Za-z]+", "", t)
        if re.search(r"\d", t):
            bad.append((n, line.strip()))
    for n, l in bad:
        print(f"body line {n}: {l[:120]}")
    print(f"{len(bad)} line(s) with raw numbers")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
