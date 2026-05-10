#!/usr/bin/env python3
"""Regenerate the built-in skills table in README.md from SKILL.md frontmatter.

Run from the repo root:
    python scripts/gen_skills_table.py

It rewrites the block between <!-- SKILLS-TABLE-START --> and
<!-- SKILLS-TABLE-END --> in README.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "stat_agent" / "skills"
README = REPO_ROOT / "README.md"

START = "<!-- SKILLS-TABLE-START -->"
END = "<!-- SKILLS-TABLE-END -->"

CATEGORIES: list[tuple[str, list[str]]] = [
    ("Cell type annotation", [
        "celltype-annotation-scANVI",
        "celltype-annotation-GPT",
        "annotation-tangram",
    ]),
    ("Spot deconvolution", [
        "celltype-deconvolution-RCTD",
        "deconvolution-cell2location",
        "deconvolution-flashdeconv",
    ]),
    ("Spatial domains", [
        "spatial-domain-SpaGCN",
        "spatial-domain-STAGATE",
        "spatial-domain-GraphST",
    ]),
    ("Spatial statistics & niches", [
        "spatial-statistics-squidpy",
        "spatial-stats-neighborhood-enrichment",
        "niche-detection-Harmonics",
        "svg-SpatialDE",
    ]),
    ("Differential expression & pathway", [
        "differential-expression",
        "pathway-GO-enrichment",
        "enrichment-ora-ssgsea",
        "pathway-ssgsea",
        "pathway-enrichment-compare",
    ]),
    ("Cell-cell communication", [
        "cell-communication-LIANA",
        "cell-communication-CellPhoneDB",
    ]),
    ("Multi-slice integration", [
        "integration-Harmony",
        "integration-bbknn",
        "integration-scanorama",
    ]),
    ("Slice alignment & registration", [
        "alignment-STalign",
        "registration-paste",
    ]),
    ("Trajectory inference", [
        "trajectory-palantir-dpt",
    ]),
]


def parse_frontmatter(text: str) -> dict:
    m = re.match(r"---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    body = m.group(1)
    out: dict = {}
    for line in body.splitlines():
        ms = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$", line)
        if ms:
            out[ms.group(1)] = ms.group(2).strip().strip('"').strip("'")
    return out


def first_sentence(text: str, max_chars: int = 140) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    cut = re.split(r"(?<=[.!?])\s", text, maxsplit=1)
    s = cut[0] if cut else text
    if len(s) > max_chars:
        s = s[: max_chars - 1].rstrip() + "…"
    return s


def load_skills() -> dict[str, dict]:
    skills: dict[str, dict] = {}
    for child in sorted(SKILLS_DIR.iterdir()):
        skill_md = child / "SKILL.md"
        if not skill_md.is_file():
            continue
        meta = parse_frontmatter(skill_md.read_text())
        if not meta:
            continue
        skills[child.name] = meta
    return skills


def render() -> str:
    skills = load_skills()
    lines: list[str] = []
    seen: set[str] = set()
    for category, slugs in CATEGORIES:
        lines.append(f"### {category}")
        lines.append("")
        lines.append("| Skill | Summary |")
        lines.append("| --- | --- |")
        for slug in slugs:
            if slug not in skills:
                lines.append(f"| `{slug}` | _missing SKILL.md_ |")
                continue
            seen.add(slug)
            meta = skills[slug]
            title = meta.get("title") or meta.get("name") or slug
            desc = first_sentence(meta.get("description", ""))
            link = f"stat_agent/skills/{slug}/SKILL.md"
            lines.append(f"| [{title}]({link}) | {desc} |")
        lines.append("")

    leftover = [s for s in skills if s not in seen]
    if leftover:
        lines.append("### Other")
        lines.append("")
        lines.append("| Skill | Summary |")
        lines.append("| --- | --- |")
        for slug in leftover:
            meta = skills[slug]
            title = meta.get("title") or meta.get("name") or slug
            desc = first_sentence(meta.get("description", ""))
            link = f"stat_agent/skills/{slug}/SKILL.md"
            lines.append(f"| [{title}]({link}) | {desc} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def update_readme(block: str) -> bool:
    text = README.read_text()
    pattern = re.compile(
        re.escape(START) + r".*?" + re.escape(END),
        re.DOTALL,
    )
    if not pattern.search(text):
        print(f"ERROR: markers {START} ... {END} not found in {README}", file=sys.stderr)
        return False
    new = pattern.sub(f"{START}\n{block}\n{END}", text)
    if new != text:
        README.write_text(new)
        print(f"Updated {README}")
    else:
        print("No changes.")
    return True


if __name__ == "__main__":
    block = render()
    if not update_readme(block):
        sys.exit(1)
