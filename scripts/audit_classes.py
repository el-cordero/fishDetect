#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fishdetect.config import load_yaml
from fishdetect.dataset import build_annotation_table, class_count_summary, load_class_counts
from fishdetect.utils.files import ensure_dir, write_csv_dicts, write_json, write_text


KNOWN_ISSUES = {
    "Acantharus": "Acanthurus",
    "Archarhinus": "Carcharhinus",
    "bivitatus": "bivittatus",
    "Lactophyrys": "Lactophrys",
    "Sparisoma viridae": "Sparisoma viride",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FishDetect class names without modifying annotations.")
    parser.add_argument("--dataset", required=True, help="Path to merged VIAME/DIVE dataset.")
    parser.add_argument("--out", default="outputs/class_audit", help="Output directory for audit files.")
    parser.add_argument("--aliases", default="configs/class_aliases.yaml", help="Optional class alias config.")
    parser.add_argument("--rare-threshold", type=int, default=5, help="Warn when a class has fewer annotations than this.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = ensure_dir(args.out)
    aliases = {}
    alias_path = Path(args.aliases)
    if alias_path.exists():
        aliases = load_yaml(alias_path).get("aliases", {})

    class_counts = load_class_counts(args.dataset)
    if class_counts:
        counts = {row["class"]: int(row.get("viame_row_count") or row.get("dive_feature_count") or 0) for row in class_counts}
    else:
        annotations, _, _ = build_annotation_table(args.dataset)
        counts = Counter(row["class_name"] for row in annotations)

    rows = []
    for class_name, count in sorted(counts.items()):
        suggestions = []
        for typo, replacement in KNOWN_ISSUES.items():
            if typo in class_name:
                suggestions.append(class_name.replace(typo, replacement))
        if class_name in aliases:
            suggestions.append(aliases[class_name])
        suggestions = sorted(set(suggestions))
        issue_type = "taxonomy_or_spelling" if suggestions else ""
        if count < args.rare_threshold:
            issue_type = "rare_class" if not issue_type else f"{issue_type};rare_class"
        rows.append(
            {
                "class_name": class_name,
                "annotation_count": count,
                "issue_type": issue_type,
                "suggested_name": "|".join(suggestions),
                "action": "review_only",
            }
        )

    write_csv_dicts(out / "class_audit.csv", rows, ["class_name", "annotation_count", "issue_type", "suggested_name", "action"])
    write_json(out / "class_audit.json", {"dataset": args.dataset, "rows": rows})
    flagged = [row for row in rows if row["issue_type"]]
    md = ["# Class Audit", "", f"Classes audited: {len(rows)}", f"Flagged for review: {len(flagged)}", ""]
    for row in flagged:
        md.append(f"- {row['class_name']} ({row['annotation_count']}): {row['issue_type']} -> {row['suggested_name'] or 'review'}")
    write_text(out / "class_audit.md", "\n".join(md) + "\n")
    print(f"Wrote class audit to {out}. Flagged {len(flagged)} classes for review.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
