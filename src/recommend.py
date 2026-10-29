"""Run grant recommendations on the real workbook or seeded synthetic data."""

import argparse
from dataclasses import asdict
from datetime import date
import json

from src.ardook_extra_synthetic_data import (
    DB_PATH as SYNTHETIC_DB,
    REFERENCE_DATE,
    load_checker as load_synthetic_checker,
    load_grants as load_synthetic_grants,
    load_profiles as load_synthetic_profiles,
)
from src.pipeline import LocalGenerator, OrganizationProfile, Pipeline
from src.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("real", "synthetic"), default="real")
    parser.add_argument("--description", help="Organization mission and program description.")
    parser.add_argument(
        "--attributes",
        default="{}",
        help='JSON organization facts, e.g. \'{"entity_type":"nonprofit","region":"US"}\'.',
    )
    parser.add_argument(
        "--profile-id",
        type=int,
        help="Use a generated synthetic profile by ID (synthetic dataset only).",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--explain", action="store_true", help="Generate advisory fit notes.")
    args = parser.parse_args()

    try:
        attributes = json.loads(args.attributes)
    except json.JSONDecodeError as error:
        parser.error(f"--attributes must be valid JSON: {error}")
    if not isinstance(attributes, dict):
        parser.error("--attributes must be a JSON object")
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    if args.dataset == "synthetic":
        as_of = args.as_of or REFERENCE_DATE
        if args.profile_id is not None:
            profiles = load_synthetic_profiles(SYNTHETIC_DB)
            selected = profiles.loc[profiles["id"] == args.profile_id]
            if selected.empty:
                parser.error(f"No synthetic profile with id {args.profile_id}")
            row = selected.iloc[0]
            profile = OrganizationProfile(row["description"], row["attributes"])
        elif args.description:
            profile = OrganizationProfile(args.description, attributes)
        else:
            parser.error("Provide --profile-id or --description for synthetic recommendations")
        grants = load_synthetic_grants(SYNTHETIC_DB)
        checker = load_synthetic_checker(SYNTHETIC_DB)
        store = VectorStore(path="vector_store_synthetic")
    else:
        as_of = args.as_of or date.today()
        if args.profile_id is not None:
            parser.error("--profile-id is available only with --dataset synthetic")
        if not args.description:
            parser.error("--description is required")
        profile = OrganizationProfile(args.description, attributes)
        grants = None
        checker = None
        store = VectorStore()

    pipeline = Pipeline(
        grants=grants,
        checker=checker,
        store=store,
        generator=LocalGenerator() if args.explain else None,
    )
    recommendations = pipeline.recommend(
        profile,
        as_of=as_of,
        top_k=args.top_k,
    )
    print(json.dumps([asdict(item) for item in recommendations], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
