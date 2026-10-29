"""Ardook extra synthetic data: fake grants and labeled fake nonprofits, kept apart from the real workbook.

Grants copy the Grant Tracker's 13 columns and its messy conventions (rolling or unverified
deadlines, mixed currencies, semicolon requirement lists) and load through src.ingest.clean_grants.
Programs are fictional and sources use example.org. Labels come from the same rules the
EligibilityChecker runs and ignore dates, which the pipeline checks separately.

Run `python -m src.ardook_extra_synthetic_data` to rebuild data/ardook_extra_synthetic_data.db.
Output is seeded, so reruns are identical.
"""

import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.ingest import COLUMNS, clean_grants
from src.pipeline import EligibilityChecker, EligibilityRule

DB_PATH = Path("data/ardook_extra_synthetic_data.db")
REFERENCE_DATE = date(2026, 10, 1)  # fixed so the data does not drift with today's date

# focus area -> (wording used by grants, wordings used by organizations)
FOCUS = {
    "youth education": (
        ["youth education and digital literacy", "STEM learning access for young people"],
        ["tutoring and digital skills training for underserved students", "after-school STEM mentoring for low-income teens"],
    ),
    "healthcare access": (
        ["community health and care access", "mental health and wellbeing services"],
        ["free clinics and care navigation for uninsured patients", "peer mental health support for isolated adults"],
    ),
    "climate and environment": (
        ["climate resilience and conservation", "clean water and sustainable agriculture"],
        ["watershed restoration and community climate adaptation", "training smallholder farmers in regenerative agriculture"],
    ),
    "civic tech and open data": (
        ["civic technology and open data", "responsible AI for public-interest use"],
        ["open data tools that help residents track local government spending", "auditing public-sector algorithms for bias"],
    ),
    "women's economic empowerment": (
        ["women's economic empowerment", "women-led ventures and entrepreneurship"],
        ["microloans and business coaching for women entrepreneurs", "mentoring women founders in technology"],
    ),
    "housing and homelessness": (
        ["affordable housing and homelessness prevention", "community development and basic needs"],
        ["rapid rehousing and eviction prevention for families", "building affordable homes with local residents"],
    ),
    "AI for social good": (
        ["AI for social good", "data and machine learning for humanitarian impact"],
        ["applying machine learning to disaster response and aid logistics", "building data platforms that help nonprofits measure impact"],
    ),
    "arts and culture": (
        ["community arts and cultural education", "heritage and creative practice"],
        ["free arts classes and public murals in neighborhoods", "preserving and teaching traditional music and crafts"],
    ),
}
ENTITIES = {  # profile vocabulary -> wording in a grant's requirement clause
    "nonprofit": "registered nonprofit",
    "for-profit": "for-profit company",
    "university": "university-affiliated team",
}
REGIONS = {  # profile vocabulary -> (grant clause, currency formats)
    "US": ("U.S.-based organization", ["${n:,}", "Up to ${n:,}", "${n:,}–${m:,}"]),
    "EU": ("based in an eligible Horizon Europe country", ["€{n:,}", "€{n:,} non-dilutive + mentoring/coaching"]),
    "UK": ("UK-registered organization", ["£{n:,}", "£{n:,} catalytic funding + 16-week programme"]),
    "India": ("operating in India", ["Up to ₹{n} lakh cash rewards"]),
    "Africa": ("operating in sub-Saharan Africa", ["${n:,}", "Up to ${n:,}"]),
}
REGION_NAME = {"US": "the U.S.", "EU": "the EU", "UK": "the UK", "India": "India", "Africa": "sub-Saharan Africa"}
FILLER = [
    "measurable social or environmental impact",
    "defined MVP or pilot",
    "letters of support",
    "audited financials",
    "application fee reported",
    "pitch or interview round",
    "multi-year sustainability plan",
]
TYPES = {  # program-name suffix -> plausible tracker "Type" values
    "Fund": ["Grant", "Grant + support", "Nonprofit technology grant", "R&D grant"],
    "Grant": ["Grant", "Microgrant", "AI impact grant", "Nonprofit technology grant"],
    "Award": ["Prize / cash awards", "Grant", "Microgrant"],
    "Prize": ["Prize / cash awards", "Pitch competition / cash prize"],
    "Challenge": ["Pitch competition / cash prize", "Prize / cash awards"],
    "Fellowship": ["Grant + fellowship"],
    "Initiative": ["Grant + support", "Equity-free accelerator", "Accelerator"],
    "Program": ["Accelerator", "Equity-free accelerator", "Grant + support"],
}
FITS = ["Excellent", "Strong", "Strong", "Moderate", "Moderate", "Weak–Moderate", "Strong if mission-aligned"]
NAME_A = ["Lantern", "Meridian", "Harbor", "Cedar", "Atlas", "Juniper", "Beacon", "Quill", "Summit", "Willow",
          "Compass", "Ember", "Kestrel", "Marigold", "Northstar", "Orchard", "Prairie", "Riverbend", "Sparrow", "Tidewater"]
NAME_B = ["Impact", "Community", "Futures", "Commons", "Bridge", "Catalyst", "Horizon", "Foundry", "Roots", "Signal"]
NAME_C = ["Fund", "Award", "Challenge", "Fellowship", "Prize", "Initiative", "Program", "Grant"]
RULE_CLAUSE = {"woman_led": "woman-led (woman founder or co-founder in top management)"}
PROFILE_REVENUE = [0, 50_000, 200_000, 400_000, 800_000, 2_000_000, 4_000_000, 12_000_000]


def _deadline(rng: random.Random) -> tuple[str, str]:
    """(deadline, current_status) in the tracker's styles."""
    roll = rng.random()
    if roll < 0.62:
        day = REFERENCE_DATE + timedelta(days=rng.randint(-150, 200))
        if day < REFERENCE_DATE:
            return day.isoformat(), rng.choice(["Closed", "Closed", "Closed for the current edition"])
        status = rng.choice(["Open as last reported", "Open as last reported", "Open round as last reported",
                             "Applications open as last reported", "Open; applications opened recently"])
        return day.isoformat(), status
    if roll < 0.72:
        return rng.choice(["Rolling", "Rolling / call-specific", "Rolling / hub-specific"]), "Open / rolling as last reported"
    if roll < 0.80:
        month = REFERENCE_DATE + timedelta(days=rng.randint(30, 200))
        return f"{month:%Y-%m} (exact deadline not published)", "Lead to monitor; exact active-call eligibility not verified"
    if roll < 0.90:
        return rng.choice(["Coming soon / not verified", "Coming soon / no open call verified"]), rng.choice(["Watchlist", "Watchlist only"])
    return rng.choice(["No active round verified", "2027 edition closed", "Program-specific"]), rng.choice(["No active round verified", "Closed / excluded", "Watchlist"])


def _amount(rng: random.Random, region: str | None, kind: str) -> str:
    if rng.random() < 0.12:
        return rng.choice(["Not verified", "Amount not published in prior verification", "Cash prize; amount not published"])
    template = rng.choice(REGIONS[region or "US"][1])
    sizes = [500, 1_000, 2_500, 5_000] if kind == "Microgrant" else [10_000, 25_000, 50_000, 75_000, 100_000, 250_000]
    n = rng.choice(sizes)
    return template.format(n=n if "lakh" not in template else rng.choice([2, 5, 10]), m=n * 2)


def _grant(rng: random.Random, name: str) -> tuple[dict, list[dict]]:
    focuses = rng.sample(list(FOCUS), 2 if rng.random() < 0.25 else 1)
    clauses, rules = [], []

    def add_rule(attribute, operator, value, clause):
        clauses.append(clause)
        rules.append({"attribute": attribute, "operator": operator, "value": value, "evidence": clause})

    region = None
    if rng.random() < 0.7:
        entities = rng.sample(list(ENTITIES), rng.choice([1, 1, 2]))
        add_rule("entity_type", "in", entities, " or ".join(ENTITIES[e] for e in entities).capitalize())
    if rng.random() < 0.55:
        region = rng.choice(list(REGIONS))
        regions = [region] + (["UK"] if region == "EU" and rng.random() < 0.5 else [])
        add_rule("region", "in", regions, REGIONS[region][0][0].upper() + REGIONS[region][0][1:])
    if rng.random() < 0.35:
        limit, label = rng.choice([(250_000, "$250k"), (500_000, "$500k"), (1_000_000, "$1M"), (5_000_000, "$5M")])
        add_rule("annual_revenue", "lte", limit, f"Under {label} annual revenue")
    if rng.random() < 0.2:
        add_rule("woman_led", "in", ["yes"], RULE_CLAUSE["woman_led"][0].upper() + RULE_CLAUSE["woman_led"][1:])
    focus_text = " and ".join(rng.choice(FOCUS[f][0]) for f in focuses)
    # Scope belongs in the requirements, as in the tracker; retrieval never sees "Why It Fits".
    clauses += [f"Supports {focus_text}"] + rng.sample(FILLER, rng.randint(1, 3))
    rng.shuffle(clauses)

    kind = rng.choice(TYPES[name.split()[-1]])
    deadline, status = _deadline(rng)
    rule_names = [r["attribute"].replace("_", " ") for r in rules]
    concern = (
        f"Must confirm {', '.join(rule_names)} requirements" if rule_names else "Eligibility largely open; confirm program scope"
    ) + rng.choice(["", "; competitive field", "; small award relative to effort", "; fee reported"])
    row = {
        "Applied?": rng.choices(["No", "Yes", "Not interested"], [0.85, 0.08, 0.07])[0],
        "Program": name,
        "Type": kind,
        "Amount / Benefit": _amount(rng, region, kind),
        "Deadline": deadline,
        "Current Status (as last reported)": status,
        "Fit": rng.choice(FITS),
        "Core Requirements": "; ".join(clauses),
        "Why It Fits Ardook Data": f"Mission overlaps with {focus_text}",
        "Key Concern / Disqualifier": concern,
        "Recommended Action": rng.choice(["Apply", "Apply if eligible", "Monitor", "Prioritize eligibility assessment", "Apply if application remains lightweight"]),
        "Official Source": f"https://example.org/synthetic/{name.lower().replace(' ', '-')}",
        "Focus": json.dumps(focuses),
    }
    return row, rules


def _profile(rng: random.Random, number: int) -> tuple[tuple, str]:
    focus = rng.choice(list(FOCUS))
    entity = rng.choices(list(ENTITIES), [0.7, 0.2, 0.1])[0]
    region = rng.choice(list(REGIONS))
    attributes = {
        "entity_type": entity,
        "region": region,
        "annual_revenue": float(rng.choice(PROFILE_REVENUE)),
        "woman_led": rng.choice(["yes", "no"]),
    }
    for key in list(attributes):  # real intake data is incomplete; missing means "unknown"
        if rng.random() < 0.10:
            del attributes[key]
    what = rng.choice(FOCUS[focus][1])
    noun = {"nonprofit": "nonprofit", "for-profit": "social enterprise", "university": "university research lab"}[entity]
    text = rng.choice(
        [
            f"A {noun} providing {what}.",
            f"We are a {noun} focused on {what}, serving communities in {REGION_NAME[region]}.",
            f"Our {noun} works on {what} and is looking for early funding to grow.",
        ]
    )
    return (number, text, json.dumps(attributes), focus), focus


def _eligible(attributes: dict, rules: list[dict]) -> bool:
    """Mirror EligibilityChecker: a rule fails only when the attribute is present and violates it."""
    for rule in rules:
        value = attributes.get(rule["attribute"])
        if value is None:
            continue
        if rule["operator"] == "in" and value not in rule["value"]:
            return False
        if rule["operator"] == "lte" and value > rule["value"]:
            return False
    return True


def build(path: Path = DB_PATH, seed: int = 166, n_grants: int = 150, n_profiles: int = 200):
    rng = random.Random(seed)
    names = [f"{a} {b} {c}" for a in NAME_A for b in NAME_B for c in NAME_C]
    names = rng.sample(names, n_grants)

    grants, rules = [], []
    for name in names:
        row, grant_rules = _grant(rng, name)
        grants.append(row)
        rules += [{"program": name, **r, "value": json.dumps(r["value"])} for r in grant_rules]
    grants.sort(key=lambda g: (g["Deadline"][:1] not in "2", g["Deadline"]))  # dated rows first, like the tracker
    for order, row in enumerate(grants, start=1):
        row["Priority / Deadline Order"] = order

    profiles, matches = [], []
    rules_by_program: dict[str, list[dict]] = {}
    for r in rules:
        rules_by_program.setdefault(r["program"], []).append({**r, "value": json.loads(r["value"])})
    for number in range(1, n_profiles + 1):
        profile, focus = _profile(rng, number)
        profiles.append(profile)
        attributes = json.loads(profile[2])
        for index, grant in enumerate(grants):
            if focus not in json.loads(grant["Focus"]):
                label = "unrelated"
            elif _eligible(attributes, rules_by_program.get(grant["Program"], [])):
                label = "match"
            else:
                label = "ineligible"
            matches.append((number, index + 2, label))  # source_row, as clean_grants numbers it

    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as db:
        frame = pd.DataFrame(grants, columns=list(COLUMNS) + ["Focus"])
        frame.to_sql("grants", db, index=False)
        pd.DataFrame(rules).to_sql("rules", db, index=False)
        db.execute("CREATE TABLE profiles (id, description, attributes, focus)")
        db.executemany("INSERT INTO profiles VALUES (?, ?, ?, ?)", profiles)
        db.execute("CREATE TABLE matches (profile_id, source_row, label)")
        db.executemany("INSERT INTO matches VALUES (?, ?, ?)", matches)


def load_grants(path: Path = DB_PATH) -> pd.DataFrame:
    """Cleaned the same way as the real tracker, so Pipeline treats both identically."""
    with sqlite3.connect(path) as db:
        return clean_grants(pd.read_sql("SELECT * FROM grants", db))


def load_checker(path: Path = DB_PATH) -> EligibilityChecker:
    with sqlite3.connect(path) as db:
        rows = pd.read_sql("SELECT * FROM rules", db)
    checker_rules: dict[str, list[EligibilityRule]] = {}
    for row in rows.itertuples():
        value = json.loads(row.value)
        checker_rules.setdefault(row.program, []).append(
            EligibilityRule(row.attribute, row.operator, value, row.evidence)
        )
    return EligibilityChecker(checker_rules)


def load_profiles(path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(path) as db:
        profiles = pd.read_sql("SELECT * FROM profiles", db)
    profiles["attributes"] = profiles["attributes"].map(json.loads)
    return profiles


def load_matches(path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(path) as db:
        return pd.read_sql("SELECT * FROM matches", db)


if __name__ == "__main__":
    build()
    print(f"Wrote {DB_PATH}")
