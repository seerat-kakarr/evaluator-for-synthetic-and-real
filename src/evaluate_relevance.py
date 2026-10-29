"""Compare grant ranking with tracker ratings or Michael's synthetic labels."""

import argparse
import hashlib
import json
import re
from datetime import date

import numpy as np
import pandas as pd

from src.ardook_extra_synthetic_data import (
    DB_PATH,
    REFERENCE_DATE,
    load_checker,
    load_grants as load_synthetic_grants,
    load_matches,
    load_profiles,
)
from src.ingest import load_grants
from src.pipeline import OrganizationProfile, Pipeline
from src.vector_store import VectorStore

K = 5
TOKEN = re.compile(r"[a-z0-9]+")
ARDOOK_PROFILE_DESCRIPTION = (
    "Ardook is an Armenia-based social enterprise that provides flexible, "
    "work-from-home laundry jobs and training to single mothers caring for "
    "children with disabilities. It offers residential and commercial laundry "
    "services, helping mothers earn income while balancing caregiving."
)


class KeywordEmbedder:
    """Create deterministic token-overlap vectors for the baseline ranker."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 4096))
        for row, text in enumerate(texts):
            for token in TOKEN.findall(text.casefold()):
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 4096
                vectors[row, index] += 1
        return vectors


class MemoryStore:
    """Use vectors in memory for evaluation; only the semantic index persists."""

    def is_current(self, ids, documents, embedder):
        return False

    def replace(self, ids, documents, vectors, embedder):
        self.ids = ids
        self.vectors = vectors

    def similarities(self, query):
        scores = (self.vectors @ query.T).ravel()
        return dict(zip(self.ids, scores.tolist()))


def _pipelines(grants, checker=None, store=None):
    return {
        "semantic": Pipeline(grants, checker=checker, store=store),
        "keyword": Pipeline(
            grants,
            checker=checker,
            embedder=KeywordEmbedder(),
            store=MemoryStore(),
        ),
    }


def _open_status(status: str) -> bool:
    status = status.casefold()
    return ("open" in status or "rolling" in status) and not any(
        phrase in status for phrase in ("no open", "not verified", "closed")
    )


def _run(pipelines, profile, as_of, relevant, labels=None):
    results = {}
    for name, pipeline in pipelines.items():
        recommendations = pipeline.recommend(
            profile,
            as_of=as_of,
            top_k=K,
            min_similarity=0.25 if name == "semantic" else 1e-12,
        )
        rows = [item.source_row for item in recommendations]
        hits = sum(row in relevant for row in rows)
        details = []
        for item in recommendations:
            row = labels.get(item.source_row) if labels else None
            details.append(
                {
                    "program": item.program,
                    "label": row,
                    "open_or_rolling": _open_status(item.evidence["current_status"]),
                    "eligibility": item.eligibility,
                    "source_url": item.source_url,
                }
            )
        results[name] = {
            "precision_at_5": hits / K,
            "recall_at_5": hits / len(relevant) if relevant else None,
            "relevant_count": len(relevant),
            "returned_count": len(recommendations),
            "open_or_rolling_count": sum(row["open_or_rolling"] for row in details),
            "known_rule_ineligible_count": (
                sum(row["label"] == "ineligible" for row in details)
                if labels and any(
                    label in {"match", "ineligible", "unrelated"}
                    for label in labels.values()
                )
                else "not_measured"
            ),
            "known_filter_errors": sum(
                any(check.status == "fail" for check in item.checks)
                for item in recommendations
            ),
            "full_eligibility_errors": "not_measured",
            "top_five": details,
        }
    return results


def _format_report(report):
    if report["dataset"] == "synthetic integration check only":
        lines = [
            "SYNTHETIC RESULTS (generated data)",
            f"Grants: {report['grants']} | Profiles: {report['profiles']} | "
            f"Reference date: {report['date']}",
            "",
            "Ranking comparison (averaged across profiles)",
            f"{'Method':<19} {'Precision@5':>12} {'Recall@5':>10} "
            f"{'Ineligible results':>19}",
        ]
        for name, title in (
            ("keyword", "Keyword baseline"),
            ("semantic", "Semantic model"),
        ):
            result = report["summary"][name]
            lines.append(
                f"{title:<19} {result['precision_at_5']:>11.1%} "
                f"{result['recall_at_5']:>9.1%} "
                f"{result['known_rule_ineligible_count']:>19}"
            )
        lines.extend(
            [
                "",
                "Note: These labels are generated using the same rules as the checker.",
            ]
        )
        return "\n".join(lines)

    lines = [
        "GRANT RANKING EVALUATION — official tracker",
        f"Grants in workbook: {report['grants']} | Evaluation date: {report['date']}",
        "",
        f"Applicant profile used: {report['profile']}",
        "",
        "What counts as a match: tracker Fit rating contains Strong or Excellent.",
        "These are team-entered ratings, not independently verified labels.",
        "Open/rolling means the tracker text says so; grant websites were not checked.",
        "Full grant eligibility is NOT measured.",
        "",
        "RANKING COMPARISON",
        f"{'Method':<19} {'Relevant / 5 (returned)':>22} {'Precision@5':>12} "
        f"{'Recall@5':>14} {'Open/rolling':>13}",
    ]
    results = report["results"]
    for name, title in (
        ("keyword", "Keyword baseline"),
        ("semantic", "Semantic model"),
    ):
        result = results[name]
        relevant_count = result["relevant_count"]
        hits = round(result["precision_at_5"] * K)
        recall = (
            f"{hits}/{relevant_count} ({result['recall_at_5']:.1%})"
            if relevant_count
            else "no labeled grants"
        )
        lines.append(
            f"{title:<19} {hits:>2}/{K} "
            f"({result['returned_count']} returned) "
            f"{result['precision_at_5']:>11.1%} "
            f"{recall:>14} "
            f"{result['open_or_rolling_count']:>9}/5"
        )

    lines.extend(
        [
            "",
            "How to read this:",
            "- Precision@5: how many of the first five match the tracker rating.",
            "- Recall@5: how many Strong/Excellent grants were found out of all available.",
            "- Open/rolling is only the spreadsheet status, not live availability.",
            "",
            "TOP FIVE RECOMMENDATIONS",
        ]
    )
    for name, title in (
        ("keyword", "Keyword baseline"),
        ("semantic", "Semantic model"),
    ):
        lines.extend(["", f"{title}:"])
        if not results[name]["top_five"]:
            lines.append("No recommendations passed the pipeline filters.")
        for index, item in enumerate(results[name]["top_five"], start=1):
            status = "marked open/rolling" if item["open_or_rolling"] else "not marked open/rolling"
            lines.append(
                f"{index}. {item['program']} — Fit: {item['label'] or 'not rated'}; "
                f"status: {status}; eligibility: {item['eligibility']}"
            )
            if item["source_url"]:
                lines.append(f"   Source: {item['source_url']}")
    return "\n".join(lines)


def evaluate_tracker():
    grants = load_grants()
    as_of = date(2026, 10, 28)
    profile = OrganizationProfile(ARDOOK_PROFILE_DESCRIPTION)
    live_candidates = [
        row
        for row in grants.to_dict("records")
        if (pd.isna(row["deadline_date"]) or row["deadline_date"].date() >= as_of)
        and not str(row["current_status"]).casefold().startswith("closed")
    ]
    relevant = {
        int(row["source_row"])
        for row in live_candidates
        if any(word in str(row["fit"]).casefold() for word in ("strong", "excellent"))
    }
    return {
        "dataset": "official tracker",
        "grants": len(grants),
        "date": as_of.isoformat(),
        "profile": profile.description,
        "labels": "Workbook Fit column: Strong/Excellent count as relevant; others do not.",
        "label_caveat": "Team-entered ratings, not independently verified eligibility labels.",
        "eligibility_caveat": "Full eligibility errors are not measured; no reviewed rules exist for the real grants.",
        "results": _run(
            _pipelines(grants, store=MemoryStore()),
            profile,
            as_of,
            relevant,
            {int(row["source_row"]): str(row["fit"]) for row in live_candidates},
        ),
    }


def evaluate_synthetic(limit: int | None = None):
    grants = load_synthetic_grants(DB_PATH)
    profiles = load_profiles(DB_PATH)
    match_rows = load_matches(DB_PATH)
    if limit:
        profiles = profiles.head(limit)
        match_rows = match_rows[match_rows["profile_id"].isin(profiles["id"])]
    labels = {
        int(profile_id): dict(zip(group.source_row.astype(int), group.label))
        for profile_id, group in match_rows.groupby("profile_id")
    }
    pipelines = _pipelines(
        grants,
        checker=load_checker(DB_PATH),
        store=VectorStore(path="vector_store_synthetic_eval"),
    )
    results = []
    for profile in profiles.itertuples(index=False):
        profile_labels = labels[int(profile.id)]
        relevant = {
            row for row, label in profile_labels.items() if label == "match"
        }
        results.append(
            _run(
                pipelines,
                OrganizationProfile(profile.description, profile.attributes),
                REFERENCE_DATE,
                relevant,
                profile_labels,
            )
        )

    summary = {}
    for name in ("semantic", "keyword"):
        system = [row[name] for row in results]
        summary[name] = {
            "precision_at_5": sum(row["precision_at_5"] for row in system) / len(system),
            "recall_at_5": sum(row["recall_at_5"] or 0 for row in system) / len(system),
            "known_rule_ineligible_count": sum(
                row["known_rule_ineligible_count"]
                for row in system
                if isinstance(row["known_rule_ineligible_count"], int)
            ),
        }
    return {
        "dataset": "synthetic integration check only",
        "grants": len(grants),
        "profiles": len(results),
        "date": REFERENCE_DATE.isoformat(),
        "caveat": "Synthetic labels are generated from the same rules as the checker; not real-world accuracy evidence.",
        "summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("tracker", "real", "synthetic"),
        default="tracker",
        help="tracker (alias: real) uses the workbook Fit ratings; synthetic uses generated labels.",
    )
    parser.add_argument("--limit-profiles", type=int)
    parser.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Output a readable report (default) or machine-readable JSON.",
    )
    args = parser.parse_args()
    if args.limit_profiles is not None and args.limit_profiles < 1:
        parser.error("--limit-profiles must be positive")
    result = (
        evaluate_synthetic(args.limit_profiles)
        if args.dataset == "synthetic"
        else evaluate_tracker()
    )
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(_format_report(result))
