"""Generate the Practical 1 training set.

The data is SYNTHETIC. It is built from templates so the run is reproducible
and the resulting JSONL stays diffable in Git. It is good enough to prove the
fine-tuning pipeline works; it is not a real compliance corpus.

Usage:  python data/make_dataset.py
"""
from __future__ import annotations

import itertools
import json
import os
import random

SEED = 42
N_EVAL = 60
PER_TEMPLATE = 11   # distinct renderings per violation template

HERE = os.path.dirname(os.path.abspath(__file__))

INSTRUCTION = (
    "Review the following financial document snippet for compliance issues. "
    "Respond with JSON only."
)

FIRMS = ["Northgate Capital", "Ellis & Roe Advisors", "Sterling Peak Wealth",
         "Halcyon Investment Partners", "Bridgeway Financial", "Cardinal Trust Group",
         "Meridian Asset Management", "Oakfield Securities", "Ravensworth Private Bank",
         "Lowell & Crane LLP", "Ashcombe Capital", "Trident Wealth Services"]
PRODUCTS = ["the Growth Portfolio", "our Balanced Fund", "the High-Yield Note",
            "the Managed Futures Account", "our Emerging Markets Fund",
            "the Structured Income Plan", "the Dynamic Allocation Fund",
            "our Private Credit Vehicle", "the Global Equity Sleeve",
            "the Capital Protected Bond"]
PERCENTS = ["8%", "12%", "15%", "9.5%", "20%", "11%", "7.25%", "18%", "6%", "14.5%"]
AMOUNTS = ["$25,000", "$50,000", "£40,000", "$120,000", "$15,000", "$250,000",
           "€75,000", "$310,000", "£18,500", "$95,000"]
PERIODS = ["the last three years", "2021-2023", "the trailing 12 months",
           "the past five years", "since inception", "2019-2024",
           "the last eight quarters", "the 2023 calendar year"]

# (category, severity, snippet templates, explanation)
VIOLATIONS = [
    (
        "misleading_performance", "high",
        [
            "{firm} guarantees a minimum return of {pct} per year on {product}.",
            "Investors in {product} are assured {pct} annual returns with no downside.",
            "{product} returned {pct} over {period}, so clients can expect the same going forward.",
            "This is a risk-free opportunity to earn {pct} through {product}.",
        ],
        "Performance is presented as guaranteed or projected without a past-performance "
        "caveat. Returns must not be described as assured or risk-free.",
    ),
    (
        "missing_risk_disclosure", "high",
        [
            "Marketing sheet for {product} describes expected gains over {period}. No risk section is present.",
            "The {product} one-pager lists benefits and target returns but contains no capital-at-risk warning.",
            "{firm} promotes {product} to retail clients; the document omits any mention of possible loss of capital.",
        ],
        "The document markets an investment product without disclosing that capital "
        "is at risk. A prominent risk warning is required.",
    ),
    (
        "fee_transparency", "medium",
        [
            "Clients pay an annual management fee. Additional charges may apply as set out separately.",
            "{firm} charges a performance fee on {product}; the rate is not stated in this document.",
            "A platform fee is deducted from returns. Exact figures available on request.",
            "Net returns shown for {product} over {period} are quoted before deduction of all applicable charges.",
        ],
        "Fees are referenced but not quantified in the document. Total costs and "
        "charges must be disclosed clearly and in full.",
    ),
    (
        "kyc_aml", "high",
        [
            "Account opened for a new client with {amount} wired from a third party. Source of funds not recorded.",
            "{amount} deposit accepted; identity verification marked as 'to follow'.",
            "Client onboarded remotely at {firm}. No proof of address or ID document is on file.",
            "A series of {amount} transfers was processed without enhanced due diligence on the beneficial owner.",
        ],
        "Customer due diligence is incomplete. Identity verification and source of "
        "funds must be established before the account is used.",
    ),
    (
        "suitability", "medium",
        [
            "{product} was recommended to a retail client with a stated low risk tolerance.",
            "A {amount} allocation to {product} was advised for a client aged 71 seeking capital preservation.",
            "{firm} recommended a leveraged position in {product} to a first-time investor.",
        ],
        "The recommended product appears inconsistent with the client's stated risk "
        "profile and objectives. A suitability assessment must support the advice.",
    ),
    (
        "data_privacy", "medium",
        [
            "Client records including full account numbers are retained indefinitely on a shared drive.",
            "{firm} shares client contact data with third-party marketing partners; no consent record is referenced.",
            "Customer identity documents are emailed unencrypted between branch offices.",
        ],
        "Personal and financial data is handled without adequate retention limits, "
        "consent, or security controls.",
    ),
    (
        "record_keeping", "low",
        [
            "Advice given to the client during a phone call on the {product} allocation was not documented.",
            "{firm} retains meeting notes for six months before deletion.",
            "The trade rationale for a {amount} position in {product} is not recorded in the client file.",
        ],
        "Required records of advice and transactions are missing or retained for too "
        "short a period.",
    ),
]

COMPLIANT = [
    "{product} carries a capital-at-risk warning: 'The value of investments may fall as well as rise and you may get back less than you invested.'",
    "Past performance for {product} over {period} is shown alongside the statement that past performance is not a reliable indicator of future results.",
    "{firm} discloses a total ongoing charge of 0.85% per year, inclusive of platform and management fees.",
    "Client identity was verified against a passport and a utility bill dated within three months; source of funds recorded as salary income.",
    "The suitability report records the client's medium risk tolerance, a five-year horizon, and explains why {product} matches both.",
    "Client data is retained for six years after the end of the relationship, in line with the firm's published retention policy.",
    "A {amount} transaction was screened against sanctions lists before settlement, and the check is logged in the client file.",
    "The advice given on {product} was documented in a written suitability letter sent to the client within five business days.",
    "{firm} states that {product} is intended for professional investors only and is not available to retail clients.",
    "All charges applying to {product}, including entry, ongoing and exit fees, are itemised in a cost table.",
]

COMPLIANT_EXPLANATION = (
    "The snippet contains the required disclosure and no compliance issue is "
    "apparent in the text provided."
)


SLOTS = {"firm": FIRMS, "product": PRODUCTS, "pct": PERCENTS,
         "amount": AMOUNTS, "period": PERIODS}


def expand(template: str, limit: int, rng: random.Random) -> list[str]:
    """Render `template` up to `limit` times, every rendering distinct.

    Iterates the cartesian product of only the slots the template actually
    uses, so a one-slot template can never collide with itself.
    """
    used = [name for name in SLOTS if "{" + name + "}" in template]
    if not used:
        return [template]

    combos = list(itertools.product(*(SLOTS[name] for name in used)))
    rng.shuffle(combos)
    return [template.format(**dict(zip(used, combo))) for combo in combos[:limit]]


def build_records() -> list[dict]:
    rng = random.Random(SEED)
    records: list[dict] = []

    # Non-compliant examples: every template rendered with distinct fills.
    for category, severity, templates, explanation in VIOLATIONS:
        for template in templates:
            for text in expand(template, PER_TEMPLATE, rng):
                records.append({
                    "instruction": INSTRUCTION,
                    "input": text,
                    "output": json.dumps({
                        "issue_found": True,
                        "category": category,
                        "severity": severity,
                        "explanation": explanation,
                    }),
                })

    # Compliant examples: roughly balanced against the violation count.
    per_clean = max(1, (len(records) * 2 // 3) // len(COMPLIANT))
    for template in COMPLIANT:
        for text in expand(template, per_clean, rng):
            records.append({
                "instruction": INSTRUCTION,
                "input": text,
                "output": json.dumps({
                    "issue_found": False,
                    "category": "none",
                    "severity": "none",
                    "explanation": COMPLIANT_EXPLANATION,
                }),
            })

    # Deduplicate on the input text; templates can collide after filling.
    seen: set[str] = set()
    unique = []
    for record in records:
        if record["input"] in seen:
            continue
        seen.add(record["input"])
        unique.append(record)

    rng.shuffle(unique)
    return unique


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {path}  ({len(records)} examples)")


def main() -> None:
    records = build_records()
    eval_records, train_records = records[:N_EVAL], records[N_EVAL:]
    write_jsonl(os.path.join(HERE, "train.jsonl"), train_records)
    write_jsonl(os.path.join(HERE, "eval.jsonl"), eval_records)

    positives = sum(1 for r in train_records if '"issue_found": true' in r["output"])
    print(f"train split: {positives} with issues / {len(train_records) - positives} clean")


if __name__ == "__main__":
    main()
