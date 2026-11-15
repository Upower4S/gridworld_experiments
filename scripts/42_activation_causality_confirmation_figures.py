#!/usr/bin/env python3
"""Render frozen confirmation figures from script40's final JSON only.

No raw-result reads, model imports, statistical recomputation, or selection.
Synthetic fixtures must be explicitly labelled and are never scientific results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import textwrap
import uuid

FORMAT = "combined_frozen_confirmation_report_v1"
CATEGORIES = ("target", "original", "third_action", "unresolved")
COLORS = ("#247C73", "#526477", "#D69A51", "#E3D7D3")
CATEGORY_LABELS = ("Physical-CF target", "Original action", "Third action", "Unresolved")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    report = json.loads(raw, object_pairs_hook=unique)
    canonical(report)
    return report


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def interval(value, label):
    require(isinstance(value, list) and len(value) == 2 and all(finite(x) for x in value)
            and -1 <= value[0] <= value[1] <= 1, "invalid " + label)
    return value


def label(value):
    return value.replace("_", " ").replace("0p5", "0.5")


def prepare(report, *, synthetic=False):
    """Validate a display view; all estimates are copied from the source report."""
    require(report.get("format") == FORMAT and report.get("mode") in {"complete_only", "at_cutoff"}, "script40 final report required")
    require((report.get("synthetic_fixture") is True) == synthetic, "synthetic fixture flag must match --synthetic")
    primary = report["primary"]
    registry = primary["confirmation_primary_family"]
    require(registry["family_size"] == 4 and len(registry["candidates"]) == 2
            and all(len(ids) == 2 and len(set(ids)) == 2 for ids in registry["candidates"].values()), "four reserved co-primary slots required")
    components = {(row["candidate_id"], row["primary_id"]): row for row in primary["primary_components"]}
    keys = [(cid, pid) for cid, ids in registry["candidates"].items() for pid in ids]
    require(len(components) == len(primary["primary_components"]) == 4 and set(components) == set(keys), "missing/extra/duplicate primary slots")
    require(primary["analysis_config"]["version"] == 4 and primary["analysis_config"]["confirmation_primary_family_size"] == 4,
            "wrong analysis/multiplicity contract")
    require(primary["analysis_config"]["minimum_effect"] == .10, "unsupported observed-effect criterion")
    effect_rows = []
    for cid, pid in keys:
        row = components[cid, pid]
        binding = registry["bindings"][cid][pid]
        if "binding" in row:
            require(canonical(row["binding"]) == canonical(binding), "component/registry binding differs")
        for name in ("treatment", "control"):
            if name in row:
                require(row[name] == binding[name], "component contrast label differs")
        state = row["component_status"]
        require(state in {"complete", "pending", "invalid"}, "invalid component status")
        require(report["mode"] != "complete_only" or state == "complete", "complete-only report contains an unestimated component")
        whole = report["whole_candidate_inventories"][cid]
        require(whole["whole_cohort_status"] in {"complete", "pending", "invalid"}, "invalid whole-cohort state")
        require(type(whole["expected_n_layouts"]) is int and whole["expected_n_layouts"] == 128, "primary layout denominator differs")
        require(state != "complete" or whole["whole_cohort_status"] == "complete", "estimate from incomplete/invalid whole cohort")
        display = {"candidate_id": cid, "primary_id": pid, "state": state, "treatment": binding["treatment"],
                   "control": binding["control"], "n_layouts": whole["expected_n_layouts"],
                   "effect": None, "adjusted_ci": None, "completion_bounds": None,
                   "source_statistical_verdict": row["statistical_verdict"]}
        if state == "complete":
            p = row["primary"]
            bounds = row["completion_sensitive_effect_bounds"]
            require(finite(p["effect"]) and -1 <= p["effect"] <= 1, "invalid ITT point estimate")
            display.update(effect=p["effect"], adjusted_ci=interval(p["adjusted_ci"], "adjusted confidence interval"),
                           completion_bounds=interval([bounds["lower"], bounds["upper"]], "completion sensitivity bounds"))
            require(display["completion_bounds"][0] <= p["effect"] <= display["completion_bounds"][1], "ITT outside completion bounds")
        # Pending/invalid slots are deliberately unestimated even if diagnostic
        # numbers are retained elsewhere in a cutoff report.
        effect_rows.append(display)
    metadata = report["cohort_display_metadata"]
    counts = report["complete_cohort_condition_counts"]
    require(set(metadata) == set(counts) and len(metadata) == 4, "all four frozen cohort inventories required")
    cohorts = []
    candidate_order = list(registry["candidates"])
    for role in ("primary", "lexical_supplement"):
        for cid in candidate_order:
            matches = [(eid, m) for eid, m in metadata.items() if m["candidate_id"] == cid and m["cohort_role"] == role]
            require(len(matches) == 1, "exactly one primary and alternate per candidate required")
            eid, m = matches[0]
            order = m["condition_order"]
            require(len(order) in {14, 15} and len(set(order)) == len(order), "invalid complete control inventory")
            require(set(m["expected_condition_trials"]) == set(order), "expected condition denominators differ")
            n = 128 if role == "primary" else 32
            require(m["expected_n_layouts"] == m["expected_n_pairs"] == n
                    and all(type(x) is int and x == n for x in m["expected_condition_trials"].values())
                    and m["expected_jobs"] == n * len(order), "fixed128/same32 denominators changed")
            require(m["native_seeds"] == [42] and m["max_new_tokens"] == 4096 and m["temperatures"] == [.7], "native seed/cap/temperature differs")
            require(m["semantic_headline_eligible"] is False, "unsupported semantic promotion")
            family = m["family_layout_counts"]
            require(family and all(type(x) is int and x > 0 for x in family.values()) and sum(family.values()) == n,
                    "planned family denominators differ")
            if role == "lexical_supplement":
                parent = metadata.get(m["parent_experiment_id"])
                require(parent is not None and parent["candidate_id"] == cid and parent["cohort_role"] == "primary", "alternate parent binding differs")
            table = counts[eid]
            if table is not None:
                if role == "primary":
                    require(report["whole_candidate_inventories"][cid]["whole_cohort_status"] == "complete", "counts from incomplete/invalid primary cohort")
                require(set(table) == set(order), "all controls must appear in completed condition counts")
                for name, row in table.items():
                    require(row["n"] == m["expected_condition_trials"][name]
                            and row["n_layouts"] == row["n_pairs"] == n
                            and all(type(row[k]) is int and row[k] >= 0 for k in CATEGORIES)
                            and sum(row[k] for k in CATEGORIES) == row["n"], "native categories/denominator inconsistent")
                    status = row["generation_status_counts"]
                    require(set(status) <= {"complete", "truncated", "malformed"}
                            and all(type(v) is int and v >= 0 for v in status.values())
                            and sum(status.values()) == row["n"]
                            and status.get("truncated", 0) + status.get("malformed", 0) == row["unresolved"], "native unresolved/status counts differ")
                    require(set(row["by_family"]) == set(family), "per-family count inventory differs")
                    for fam, values in row["by_family"].items():
                        require(values["n"] == values["n_layouts"] == values["n_pairs"] == family[fam]
                                and all(type(values[k]) is int and values[k] >= 0 for k in CATEGORIES)
                                and sum(values[k] for k in CATEGORIES) == values["n"], "per-family category/denominator inconsistent")
                        family_status = values["generation_status_counts"]
                        require(set(family_status) <= {"complete", "truncated", "malformed"}
                                and all(type(v) is int and v >= 0 for v in family_status.values())
                                and sum(family_status.values()) == values["n"]
                                and family_status.get("truncated", 0) + family_status.get("malformed", 0) == values["unresolved"],
                                "per-family native status counts differ")
                    require(all(sum(v["generation_status_counts"].get(k, 0) for v in row["by_family"].values()) == status.get(k, 0)
                                for k in {"complete", "truncated", "malformed"}), "per-family statuses do not sum to complete control")
                    require(all(sum(values[k] for values in row["by_family"].values()) == row[k] for k in CATEGORIES),
                            "per-family counts do not sum to complete control")
            elif report["mode"] == "complete_only":
                raise ValueError("complete-only report omits a full cohort")
            cohorts.append({"experiment_id": eid, "metadata": deepcopy(m), "counts": deepcopy(table)})
    return {"format": "confirmation_display_counts_v1", "freeze_id": report["freeze_id"], "report_mode": report["mode"],
            "synthetic_fixture": synthetic, "primary_effects": effect_rows, "cohorts": cohorts,
            "candidate_conjunctions": deepcopy(primary["candidate_conjunctions"]),
            "reserved_primary_slots": 4, "statistical_recomputation": False,
            "bar_estimand": "unweighted observed trial fractions; primary effects use frozen family weights",
            "completion_bounds_scope": "observed-cohort sensitivity bounds, not confidence intervals",
            "interpretation": "Both co-primary components are required per candidate. Alternates repeat the same 32 layouts; no equivalence or semantic claim."}


def figures(view):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    synthetic = "SYNTHETIC FIXTURE — NOT RESULTS\n" if view["synthetic_fixture"] else ""
    fig, ax = plt.subplots(figsize=(14, 6.4))
    fig.subplots_adjust(left=.31, right=.87, top=.77, bottom=.30)
    ax.axvline(0, color="#8B9398", lw=.8)
    ax.axvline(10, color="#7C6D7E", lw=.8, ls=":")
    for index, row in enumerate(view["primary_effects"]):
        if row["state"] == "complete":
            ci, bounds = row["adjusted_ci"], row["completion_bounds"]
            ax.plot([100*x for x in ci], [index-.09]*2, color="#28618B", lw=4, solid_capstyle="butt")
            ax.plot([100*x for x in bounds], [index+.12]*2, color="#C18131", lw=2, ls="--", marker="|", ms=9)
            ax.scatter([100*row["effect"]], [index-.09], color="#152F42", marker="D", s=36, zorder=4)
            text = f"{100*row['effect']:+.1f} pp"
        else:
            ax.axhspan(index-.28, index+.28, color="#F0F1F2", zorder=0)
            ax.text(0, index, "UNESTIMATED · " + row["state"].upper(), va="center", ha="center", fontsize=9, color="#667078")
            text = "—"
        ax.text(1.035, index, text, transform=ax.get_yaxis_transform(), va="center", fontsize=11)
    labels = [f"{label(r['candidate_id'])} · N={r['n_layouts']}\n{label(r['treatment'])} − {label(r['control'])}" for r in view["primary_effects"]]
    ax.set_yticks(range(4), [textwrap.fill(v, 49, replace_whitespace=False) for v in labels], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_ylim(3.55, -.55)
    ax.set_xlim(-104, 104)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xlabel("Native target-action rate difference (percentage points)", labelpad=9)
    fig.suptitle(synthetic + "Frozen confirmation: four co-primary contrasts", x=.03, ha="left", y=.98, fontsize=17, fontweight="bold")
    fig.text(.03, .835, "Native seed 42 · cap 4,096 · all four multiplicity slots retained · both components required per candidate", fontsize=10, color="#52616B")
    handles = [Line2D([0], [0], marker="D", color="none", markerfacecolor="#152F42", label="ITT point"),
               Line2D([0], [0], color="#28618B", lw=4, label="Adjusted simultaneous CI"),
               Line2D([0], [0], color="#C18131", lw=2, ls="--", label="Completion sensitivity bounds")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.025, .15), ncol=3, frameon=False, fontsize=10)
    caption = ("Blue intervals are copied from the joint adjusted analysis. Orange bounds vary unresolved completions within the observed cohort; they are not sampling intervals. "
               "The dotted line marks the prespecified observed-effect criterion of +10 pp. An interval above zero does not establish a population effect of at least 10 pp. "
               "Unresolved native outputs remain zero target success in ITT; missing or technically invalid slots remain unestimated. Decision rules and review evidence remain in the source report.")
    fig.text(.03, .125, textwrap.fill(caption, 163), ha="left", va="top", fontsize=9, linespacing=1.45, color="#46515A")

    bars, axes = plt.subplots(2, 2, figsize=(19.5, 15.5))
    bars.subplots_adjust(left=.225, right=.89, top=.875, bottom=.105, wspace=.94, hspace=.37)
    bars.suptitle(synthetic + "All frozen controls: native action outcomes", x=.03, ha="left", y=.985, fontsize=19, fontweight="bold")
    bars.text(.03, .93, "Observed trial fractions · target / original / third / unresolved counts · seed 42 · native cap 4,096", fontsize=11, color="#52616B")
    # Primary candidates occupy the first row; their same-layout alternates the second.
    for ax, cohort in zip(axes.flat, view["cohorts"], strict=True):
        m, counts = cohort["metadata"], cohort["counts"]
        order, n = m["condition_order"], m["expected_n_layouts"]
        for index, name in enumerate(order):
            if counts is None:
                ax.barh(index, 1, height=.66, facecolor="#F5F5F5", edgecolor="#DADDE0", linewidth=.6)
                detail = "—"
            else:
                row, left = counts[name], 0
                for category, color in zip(CATEGORIES, COLORS, strict=True):
                    width = row[category] / row["n"]
                    ax.barh(index, width, left=left, height=.66, color=color, edgecolor="white", linewidth=.55,
                            hatch="////" if category == "unresolved" else None)
                    left += width
                detail = "/".join(str(row[k]) for k in CATEGORIES)
            ax.text(1.025, index, detail, transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        if counts is None:
            ax.text(.5, .5, "UNESTIMATED\nMissing or technically invalid cohort", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="#667078", bbox={"facecolor":"white", "edgecolor":"none", "alpha":.94})
        ax.set_yticks(range(len(order)), [textwrap.fill(label(name), 32) for name in order], fontsize=8)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.set_ylim(len(order)-.3, -.8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        repeat = "fresh layouts" if m["cohort_role"] == "primary" else "same layouts repeated"
        family = ", ".join(f"{label(k)} {v}" for k, v in m["family_layout_counts"].items())
        geometry = f"layer {m['layer']}" + (f" · rank {m['rank']}" if m.get("rank") is not None else "")
        ax.set_title(f"{label(m['candidate_id'])} · {m['cohort_role'].replace('_', ' ')}\n{geometry}; {n} {repeat}; {m['expected_jobs']:,} trials\n{family}",
                     loc="left", pad=16, fontsize=11, linespacing=1.5)
        ax.text(1.025, 1.01, "T / O / X / U", transform=ax.transAxes, fontsize=8, ha="left")
    handles = [Patch(facecolor=color, edgecolor="white", hatch="////" if k == "unresolved" else None, label=name)
               for k, color, name in zip(CATEGORIES, COLORS, CATEGORY_LABELS, strict=True)]
    bars.legend(handles=handles, loc="lower left", bbox_to_anchor=(.025, .065), ncol=4, frameon=False, fontsize=11)
    caption = ("Bars show unweighted counts divided by all planned trials in each complete control, including truncated/malformed native continuations. "
               "Primary co-primary effects instead use the frozen family weights. Each alternate repeats the same 32 layouts from its primary; repetitions are not additional environments. "
               "No forced-score fallback, per-condition significance test, alternate equivalence test, or semantic-belief claim is made.")
    bars.text(.03, .055, textwrap.fill(caption, 205), va="top", fontsize=10, color="#46515A", linespacing=1.5)
    return {"primary": fig, "conditions": bars}


def render(report_path, output_stem, *, synthetic=False):
    path, stem = Path(report_path).absolute(), Path(output_stem).absolute()
    require(path.resolve(strict=True) == path and stem.parent.resolve() == stem.parent, "canonical source/output paths required")
    require(not stem.is_relative_to(path.parent) or stem.parent.name in {"figures", "tmp"} or "tmp" in stem.parts,
            "write figures outside the source-report directory")
    # Outputs may be in figures, a synthetic tmp directory, or an external directory;
    # they may never be placed in a campaign input tree.
    require(not any(part in {"results", "specs", "datasets", "bases", "freezes", "planning", "source_snapshots", "leases"}
                    for part in stem.parts), "output is inside a protected input tree")
    raw = path.read_bytes()
    source_path = Path(__file__).resolve()
    source_bytes = source_path.read_bytes()
    report = strict(raw)
    view = prepare(report, synthetic=synthetic)
    if synthetic:
        require("tmp" in stem.parts or "tmp" in stem.parent.name, "synthetic figures must stay in a labelled temporary directory")
    outputs = {f"{kind}.{extension}": Path(str(stem) + f"_{kind}.{extension}")
               for kind in ("primary", "conditions") for extension in ("png", "pdf")}
    count_path = Path(str(stem) + ".counts.json")
    require(not any(p.exists() for p in [*outputs.values(), count_path]), "immutable figure output already exists")
    produced = figures(view)
    payloads = {}
    import matplotlib.pyplot as plt
    try:
        for key in outputs:
            kind, extension = key.split(".")
            stream = io.BytesIO()
            produced[kind].savefig(stream, format=extension, dpi=180, facecolor="white")
            payloads[outputs[key]] = stream.getvalue()
    finally:
        for fig in produced.values():
            plt.close(fig)
    require(path.read_bytes() == raw, "source report changed while rendering")
    require(source_path.read_bytes() == source_bytes, "renderer source changed while rendering")
    view.update(created_at=datetime.now(timezone.utc).isoformat(), source_report={"path": str(path), "sha256": digest(raw)},
                source_report_provenance=deepcopy(report["provenance"]),
                renderer={"path": str(source_path), "sha256": digest(source_bytes)},
                artifacts={str(p): digest(data) for p, data in payloads.items()},
                source_access="Only the supplied report JSON was opened; raw/source paths in its provenance were not reopened.")
    payloads[count_path] = (json.dumps(view, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Render in memory, then exclusive publication. A partial publication remains
    # visible evidence; it is never overwritten or silently repaired on rerun.
    for destination, data in payloads.items():
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {str(p): digest(data) for p, data in payloads.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output filename stem")
    parser.add_argument("--synthetic", action="store_true", help="label fixture figures, permitted only in temporary paths")
    args = parser.parse_args(argv)
    print(json.dumps(render(args.report, args.output, synthetic=args.synthetic), indent=2))


if __name__ == "__main__":
    main()
