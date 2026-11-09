#!/usr/bin/env python3
"""Build the submission files in upload/: one flattened manuscript .tex, the supplement PDF,
figures, class and style files.

Usage: python scripts/build_upload.py --doi 10.5281/zenodo.NNNNNNN
       python scripts/build_upload.py --dry-run     (no DOI; keeps the draft switch, for testing)

Steps: compile supplementary.tex and manuscript.tex; inline numbers.tex, the generated tables and
the bibliography (.bbl); replace \\ref{S-...} by the supplement's numbers; set \\draftfalse and the
DOI; compile the result in upload/ and fail on any \\todo, undefined reference or overfull box
above 5 pt.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upload"
NAME = "manuscript_upload"


def latexmk(tex, cwd):
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex],
                       cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"latexmk failed on {tex}:\n{r.stdout[-3000:]}")


def labels(aux):
    return dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux.read_text()))


def inline_inputs(s):
    def one(m):
        name = m.group(2)
        path = ROOT / ("paper/generated/" + name if m.group(1) == "geninput" else name)
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        return inline_inputs(path.read_text().rstrip("\n"))
    return re.sub(r"\\(input|geninput)\{([^}]*)\}", one, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.doi and not a.dry_run:
        sys.exit("give --doi (or --dry-run)")

    latexmk("supplementary.tex", ROOT)
    latexmk("manuscript.tex", ROOT)
    supp = labels(ROOT / "supplementary.aux")

    s = (ROOT / "manuscript.tex").read_text()
    s = s.replace("\\newcommand{\\geninput}[1]{\\input{paper/generated/#1}}\n", "")
    s = inline_inputs(s)
    s = re.sub(r"% References to the supplement.*?\\externaldocument\[S-\]\{supplementary\}\n", "", s, flags=re.S)

    def sref(m):
        if m.group(1) not in supp:
            sys.exit(f"supplement label not found: {m.group(1)}")
        return supp[m.group(1)]
    s = re.sub(r"\\ref\{S-([^}]*)\}", sref, s)
    s = s.replace("\\bibliography{references}", (ROOT / "manuscript.bbl").read_text().rstrip("\n"))
    s = s.replace("\\graphicspath{{paper/figures/}}", "\\graphicspath{{./}}")
    # Drop comment text (it names files and scripts); keep the % so line joins are unchanged
    s = "\n".join(re.sub(r"(?<!\\)%.*$", "%", line) for line in s.split("\n"))
    if a.doi:
        s = s.replace("\\drafttrue", "\\draftfalse")
        s = re.sub(r"\\newcommand\{\\ZenodoDOI\}\{.*\}\n",
                   lambda m: "\\newcommand{\\ZenodoDOI}{\\url{https://doi.org/" + a.doi + "}}\n", s)
    body = s[s.index("\\begin{document}"):]
    todos = re.findall(r"\\todo\{", body) + (re.findall(r"\\ZenodoDOI", body) if not a.doi else [])
    if todos and not a.dry_run:
        sys.exit(f"{len(todos)} \\todo left in the body")

    if UP.exists():
        shutil.rmtree(UP)
    UP.mkdir()
    (UP / f"{NAME}.tex").write_text(s)
    for f in ("sn-jnl.cls", "sn-mathphys-num.bst"):
        shutil.copy(ROOT / f, UP / f)
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body):
        shutil.copy(ROOT / "paper/figures" / fig, UP / fig)
    shutil.copy(ROOT / "supplementary.pdf", UP / "supplementary.pdf")
    latexmk(f"{NAME}.tex", UP)

    log = (UP / f"{NAME}.log").read_text(errors="replace")
    over = [float(x) for x in re.findall(r"Overfull \\[hv]box \(([\d.]+)pt too", log)]
    undef = re.findall(r"(?:Reference|Citation) `[^']*' .*undefined", log)
    pages = re.search(r"Output written on .*?\((\d+) pages", log).group(1)
    print(f"upload/{NAME}.pdf: {pages} pages, {len(undef)} undefined, "
          f"{sum(o > 5 for o in over)} overfull > 5pt, {len(todos)} todo")
    if undef or any(o > 5 for o in over):
        sys.exit(1)
    for f in UP.iterdir():
        if f.suffix in {".aux", ".log", ".fls", ".fdb_latexmk", ".out", ".blg"}:
            f.unlink()


if __name__ == "__main__":
    main()
