"""Load and clean the Ardook grant tracker without changing the workbook."""

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = {
    "Priority / Deadline Order": "priority_order",
    "Applied?": "applied",
    "Program": "program",
    "Type": "type",
    "Amount / Benefit": "amount_benefit",
    "Deadline": "deadline",
    "Current Status (as last reported)": "current_status",
    "Fit": "fit",
    "Core Requirements": "core_requirements",
    "Why It Fits Ardook Data": "why_it_fits",
    "Key Concern / Disqualifier": "concerns",
    "Recommended Action": "recommended_action",
    "Official Source": "official_source",
}


def clean_grants(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize tracker fields; retain source text and Excel row references."""
    grants = raw.copy()
    grants.columns = grants.columns.str.strip()
    missing = COLUMNS.keys() - set(grants.columns)
    if missing:
        raise ValueError(f"Missing tracker columns: {', '.join(sorted(missing))}")
    grants = grants.rename(columns=COLUMNS)
    grants["source_row"] = range(2, len(grants) + 2)

    for column in COLUMNS.values():
        grants[column] = (
            grants[column].astype("string").str.replace(r"\s+", " ", regex=True)
            .str.strip().replace("", pd.NA)
        )
    # Formatting-only rows are common in the supplied workbook.
    grants = grants.dropna(subset=list(COLUMNS.values()), how="all").copy()
    if grants["program"].isna().any():
        rows = grants.loc[grants["program"].isna(), "source_row"].tolist()
        raise ValueError(f"Nonempty tracker rows missing a program: {rows}")

    grants["priority_order"] = pd.to_numeric(
        grants["priority_order"], errors="raise"
    ).astype("Int64")
    grants["applied_bool"] = (
        grants["applied"].str.casefold().map({"yes": True, "no": False})
        .astype("boolean")
    )
    # Do not guess dates for rolling, partial, or unverified deadlines.
    exact_dates = grants["deadline"].str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    grants["deadline_date"] = pd.to_datetime(
        grants["deadline"].where(exact_dates), format="%Y-%m-%d", errors="coerce"
    )
    return grants.reset_index(drop=True)


def load_grants(path: str | Path = "Ardook_data.xlsx") -> pd.DataFrame:
    """Read only Grant Tracker; Summary contains aggregates, not opportunities."""
    raw = pd.read_excel(
        path, sheet_name="Grant Tracker", engine="openpyxl", keep_default_na=False
    )
    return clean_grants(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", nargs="?", default="Ardook_data.xlsx")
    args = parser.parse_args()
    data = load_grants(args.workbook)
    print(f"Loaded {len(data)} opportunities; {data.deadline_date.notna().sum()} exact deadlines.")
    print(data[["program", "applied_bool", "deadline", "fit"]].to_string(index=False))
