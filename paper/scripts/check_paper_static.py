#!/usr/bin/env python3
"""Static consistency checks for the paper draft.

This complements check_audited_numbers.py. It does not compile LaTeX; instead it
checks the issues that have repeatedly caused paper drift in this repo: missing
citation keys, missing refs, broken figure paths, TODO markers in included
sections, and over-strong claims that are not supported by the current evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "paper"
MAIN_TEX = PAPER_ROOT / "acl_latex.tex"

FORBIDDEN_PATTERNS = {
    "state-of-the-art mitigation": "Do not frame the current result as SOTA mitigation.",
    "solves hallucination": "The current result is a modest verifier, not a solution.",
    "solve hallucination": "The current result is a modest verifier, not a solution.",
    "external detector is the contribution": "OWLv2 is a backend, not the contribution.",
    "attention is useless": "The supported claim is proxy failure, not that attention is useless.",
    "standalone mitigator": "LH-Shape/TDEV-lite are triage/readout evidence, not standalone mitigation.",
    "Detection and Mitigation}": "The paper title should not imply a strong mitigation paper.",
    "two-stage POPE gate": "The current main POPE operating point is hybrid gate-plus-rescue.",
}

ALLOWED_TODO_FILES = {MAIN_TEX}


def _included_tex_files() -> list[Path]:
    text = MAIN_TEX.read_text()
    files = [MAIN_TEX]
    for match in re.finditer(r"\\input\{([^}]+)\}", text):
        rel = match.group(1)
        path = PAPER_ROOT / f"{rel}.tex"
        if path.exists():
            files.append(path)
    files.extend(sorted((PAPER_ROOT / "tables").glob("*.tex")))
    return files


def _paper_text(files: list[Path]) -> str:
    return "\n".join(path.read_text() for path in files)


def check_citations(files: list[Path]) -> None:
    text = _paper_text(files)
    cite_keys: set[str] = set()
    for match in re.finditer(r"\\cite[talp]?\{([^}]+)\}", text):
        cite_keys.update(key.strip() for key in match.group(1).split(",") if key.strip())

    bib_text = (PAPER_ROOT / "custom.bib").read_text()
    anthology = PAPER_ROOT / "anthology.bib.txt"
    if anthology.exists():
        bib_text += "\n" + anthology.read_text()
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    missing = sorted(cite_keys - bib_keys)
    if missing:
        raise AssertionError(f"Missing bib entries for citation keys: {missing}")
    print(f"citation keys ok: {len(cite_keys)}")


def check_refs(files: list[Path]) -> None:
    text = _paper_text(files)
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:ref|pageref)\{([^}]+)\}", text))
    missing = sorted(refs - labels)
    if missing:
        raise AssertionError(f"Missing labels for refs: {missing}")
    print(f"refs ok: {len(refs)}")


def check_figures(files: list[Path]) -> None:
    missing: list[str] = []
    for path in files:
        text = path.read_text()
        for match in re.finditer(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            raw = match.group(1)
            # LaTeX resolves graphics paths relative to the main file's
            # working directory, not the directory of the included section.
            candidate = (PAPER_ROOT / raw).resolve()
            if not candidate.exists():
                missing.append(f"{path.relative_to(PROJECT_ROOT)} -> {raw}")
    if missing:
        raise AssertionError(f"Missing figure files: {missing}")
    print("figure paths ok")


def check_todos(files: list[Path]) -> None:
    offenders = []
    for path in files:
        if path in ALLOWED_TODO_FILES:
            continue
        content = path.read_text()
        if re.search(r"\\TODO\{|TODO:", content):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    if offenders:
        raise AssertionError(f"TODO markers remain in paper content: {offenders}")
    print("TODO markers ok")


def check_forbidden_claims(files: list[Path]) -> None:
    text = _paper_text(files).lower()
    hits = []
    for pattern, reason in FORBIDDEN_PATTERNS.items():
        if pattern.lower() in text:
            hits.append(f"{pattern!r}: {reason}")
    if hits:
        raise AssertionError("Forbidden or over-strong claims found: " + "; ".join(hits))
    print("claim wording ok")


def main() -> None:
    files = _included_tex_files()
    check_citations(files)
    check_refs(files)
    check_figures(files)
    check_todos(files)
    check_forbidden_claims(files)
    print("All static paper checks passed.")


if __name__ == "__main__":
    main()
