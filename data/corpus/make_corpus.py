"""Generate the synthetic, versioned regulatory corpus.

The corpus is illustrative and modelled on the *shape* of Indian financial
regulation (RBI / SEBI / IRDAI). It is not a reproduction of any real
regulation and must not be used as legal or compliance advice.

Usage:
    python data/corpus/make_corpus.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "regulations.jsonl")


def _record(
    doc_id: str,
    title: str,
    authority: str,
    section: str,
    category: str,
    text: str,
    effective_date: str,
    last_updated: str,
    superseded_by: str | None = None,
    superseded_date: str | None = None,
) -> dict:
    return {
        "doc_id": doc_id,
        "title": title,
        "authority": authority,
        "section": section,
        "category": category,
        "text": " ".join(text.split()),
        "effective_date": effective_date,
        "last_updated": last_updated,
        "superseded_by": superseded_by,
        "superseded_date": superseded_date,
        "status": "superseded" if superseded_by else "current",
        "url": None,
        "source": "synthetic",
    }


RECORDS = [
    _record(
        "sebi-advert-2021", "Advertising Code for Intermediaries", "SEBI",
        "4.2", "misleading_performance",
        "Advertisements and client communications must not state or imply that "
        "returns are guaranteed, assured or risk-free. Any reference to past "
        "performance must be accompanied by the statement that past performance "
        "is not a reliable indicator of future results.",
        "2021-01-01", "2024-05-31",
    ),
    _record(
        "rbi-deposit-adv-2019", "Guidelines on Deposit Product Advertising", "RBI",
        "2.4", "misleading_performance",
        "Regulated entities shall not describe deposit or investment products as "
        "risk-free. Claims about returns must be accurate, balanced and supported "
        "by the terms of the product.",
        "2019-07-15", "2024-02-10",
    ),
    _record(
        "sebi-research-2020", "Research Analyst Communication Standards", "SEBI",
        "7.1", "misleading_performance",
        "Forward-looking statements in research reports must be clearly labelled "
        "as estimates or projections and must disclose the material assumptions "
        "on which they are based.",
        "2020-06-01", "2023-12-20",
    ),
    _record(
        "irdai-adv-2021", "Insurance Product Communication Guidelines", "IRDAI",
        "5.3", "misleading_performance",
        "Insurers and distributors shall not present indicative benefit "
        "illustrations as guaranteed outcomes. Illustrations must be shown at "
        "the prescribed rates together with a clear statement that benefits are "
        "not guaranteed.",
        "2021-04-01", "2024-01-15",
    ),
    _record(
        "sebi-past-perf-2022", "Disclosure of Past Performance", "SEBI",
        "3.6", "misleading_performance",
        "Where past performance is displayed, the scheme or portfolio, the period "
        "covered, the benchmark and the warning that past performance is not "
        "indicative of future results must all be disclosed with equal prominence.",
        "2022-03-01", "2024-04-05",
    ),
    _record(
        "sebi-offer-doc-2020", "Risk Disclosure in Offer Documents", "SEBI",
        "12.1", "missing_risk_disclosure",
        "Offer documents and product notes must prominently disclose the risk "
        "factors, including the possibility of loss of capital. Risk disclosures "
        "must not be buried in footnotes or omitted from marketing summaries.",
        "2020-01-15", "2024-06-01",
    ),
    _record(
        "sebi-riskometer-2021", "Product Riskometer Requirements", "SEBI",
        "6.4", "missing_risk_disclosure",
        "Product literature for mutual funds and portfolio strategies must "
        "display the prescribed riskometer and a capital-at-risk warning on the "
        "front page of the document.",
        "2021-09-01", "2024-03-22",
    ),
    _record(
        "rbi-structured-2020", "Structured Product Suitability and Disclosure", "RBI",
        "8.2", "missing_risk_disclosure",
        "Marketing material for structured products must state that capital is "
        "at risk and explain any conditions that could reduce or eliminate "
        "returns, including early redemption and counterparty risks.",
        "2020-11-10", "2023-10-18",
    ),
    _record(
        "irdai-risk-2021", "Insurance Product Risk Disclosure", "IRDAI",
        "4.8", "missing_risk_disclosure",
        "Product summaries must disclose surrender value risk, lapse "
        "consequences and any charges that reduce the maturity benefit. The "
        "disclosure must be in plain language and in the same document as the "
        "benefit illustration.",
        "2021-02-01", "2024-02-28",
    ),
    _record(
        "sebi-mf-2022", "Mutual Fund Scheme Risk Statement", "SEBI",
        "2.9", "missing_risk_disclosure",
        "All scheme advertisements and one-pagers must carry the statement that "
        "investments are subject to market risks and that investors should read "
        "all scheme-related documents carefully before investing.",
        "2022-07-01", "2024-05-12",
    ),
    _record(
        "sebi-fees-2018", "Disclosure of Total Expense Ratio", "SEBI",
        "3.3", "fee_transparency",
        "Intermediaries must disclose the total expense ratio and all recurring "
        "charges applicable to a product in the client communication. Indirect "
        "charges must be identified as such.",
        "2018-04-01", "2022-09-30",
        superseded_by="sebi-fees-2022", superseded_date="2022-10-01",
    ),
    _record(
        "sebi-fees-2022", "Total Cost Disclosure for Investment Products", "SEBI",
        "5.1", "fee_transparency",
        "A complete, itemised table of all entry, ongoing and exit charges must "
        "be provided to the client before the investment is made. The table must "
        "state each charge as an amount and as a percentage of the investment.",
        "2022-10-01", "2024-06-15",
    ),
    _record(
        "sebi-performance-fee-2021", "Performance Fee Disclosure", "SEBI",
        "4.5", "fee_transparency",
        "Where a performance fee applies, the rate, the calculation method, the "
        "benchmark and the frequency of crystallisation must be disclosed in the "
        "product document. Fees may not be described as 'available on request'.",
        "2021-06-01", "2023-11-30",
    ),
    _record(
        "rbi-charges-2020", "Transparent Customer Charges", "RBI",
        "2.7", "fee_transparency",
        "Regulated entities must publish a tariff schedule listing all charges "
        "applicable to customer accounts and must not levy any charge that is "
        "not disclosed in that schedule.",
        "2020-03-01", "2024-01-20",
    ),
    _record(
        "sebi-trail-2019", "Disclosure of Distribution Commission", "SEBI",
        "6.2", "fee_transparency",
        "Distribution commissions and trail fees paid to intermediaries must be "
        "disclosed to the client in monetary terms at the time of the "
        "recommendation and in the periodic account statement.",
        "2019-10-01", "2023-09-14",
    ),
    _record(
        "irdai-charges-2021", "Insurance Charge Disclosure", "IRDAI",
        "7.4", "fee_transparency",
        "Insurers must show every charge deducted from the policy fund, "
        "including allocation, administration, mortality and surrender charges, "
        "in the policy document and in the annual statement.",
        "2021-08-01", "2024-03-08",
    ),
    _record(
        "rbi-kyc-md-2016", "Master Direction - Know Your Customer", "RBI",
        "3.2", "kyc_aml",
        "Before establishing a business relationship, a regulated entity shall "
        "complete customer due diligence, including verification of identity and "
        "source of funds, and shall maintain the records of that verification.",
        "2016-02-25", "2023-11-06",
        superseded_by="rbi-kyc-md-2023", superseded_date="2023-11-07",
    ),
    _record(
        "rbi-kyc-md-2023", "Master Direction - Know Your Customer (Revised)", "RBI",
        "4.1", "kyc_aml",
        "Regulated entities must verify identity using a valid officially valid "
        "document and record the source of funds before activating an account. "
        "Enhanced due diligence is required for high-risk customers, politically "
        "exposed persons and beneficial owners.",
        "2023-11-07", "2024-06-20",
    ),
    _record(
        "rbi-beneficial-2022", "Beneficial Ownership Identification", "RBI",
        "5.3", "kyc_aml",
        "Regulated entities must identify the beneficial owner of every account "
        "and verify that ownership before processing transactions above the "
        "prescribed threshold.",
        "2022-05-01", "2024-04-18",
    ),
    _record(
        "rbi-pep-2022", "Politically Exposed Person Screening", "RBI",
        "6.1", "kyc_aml",
        "Customers and beneficial owners must be screened against politically "
        "exposed person and sanctions lists at onboarding and on an ongoing "
        "basis. Screening results must be retained in the client file.",
        "2022-05-01", "2024-05-09",
    ),
    _record(
        "rbi-source-funds-2021", "Source of Funds for Third-Party Transfers", "RBI",
        "5.8", "kyc_aml",
        "Where funds are received from a third party, the regulated entity must "
        "obtain and verify the source of funds and record the relationship "
        "between the account holder and the remitter before crediting the "
        "account.",
        "2021-01-01", "2023-12-01",
    ),
    _record(
        "sebi-kyc-2020", "KYC Requirements for Securities Market Intermediaries", "SEBI",
        "3.4", "kyc_aml",
        "Intermediaries must complete know-your-customer verification before "
        "executing the first transaction and must not defer identity verification "
        "until after funds have been accepted.",
        "2020-08-01", "2024-02-16",
    ),
    _record(
        "sebi-suitability-2020", "Suitability and Appropriateness Assessment", "SEBI",
        "8.1", "suitability",
        "Before recommending a product, an intermediary must assess the client's "
        "risk tolerance, investment horizon and objectives, and must record why "
        "the recommendation is suitable for that profile.",
        "2020-04-01", "2024-03-11",
    ),
    _record(
        "sebi-risk-profiling-2021", "Client Risk Profiling Standards", "SEBI",
        "4.7", "suitability",
        "Risk profiling must use an objective questionnaire, be reviewed at "
        "least annually and be updated when the client's circumstances change. "
        "A recommendation that conflicts with the recorded profile must be "
        "explained in writing.",
        "2021-07-01", "2024-01-25",
    ),
    _record(
        "rbi-deposit-suitability-2019", "Suitability of Deposit and Investment Products", "RBI",
        "2.9", "suitability",
        "Regulated entities must consider the customer's risk profile and "
        "investment objective before marketing complex or market-linked "
        "products and must not recommend such products to customers seeking "
        "capital preservation.",
        "2019-12-01", "2023-08-30",
    ),
    _record(
        "irdai-suitability-2021", "Insurance Product Suitability", "IRDAI",
        "6.6", "suitability",
        "Distributors must assess the prospect's needs, affordability and "
        "existing coverage before recommending a policy. The suitability "
        "assessment must be documented and retained.",
        "2021-03-01", "2024-04-01",
    ),
    _record(
        "sebi-leverage-2022", "Leveraged and Complex Product Restrictions", "SEBI",
        "9.3", "suitability",
        "Leveraged, derivative-linked and other complex products must not be "
        "recommended to first-time investors or to clients whose recorded risk "
        "tolerance is low, unless the client has received independent advice.",
        "2022-01-01", "2024-05-20",
    ),
    _record(
        "rbi-dp-2019", "Data Retention and Customer Consent", "RBI",
        "3.5", "data_privacy",
        "Regulated entities shall collect only the customer data necessary for "
        "the relationship, obtain consent for sharing it with third parties and "
        "retain it only for the period required by law.",
        "2019-09-01", "2023-07-31",
        superseded_by="rbi-dp-2023", superseded_date="2023-08-01",
    ),
    _record(
        "rbi-dp-2023", "Data Protection and Customer Consent (Revised)", "RBI",
        "4.3", "data_privacy",
        "Customer data must be encrypted in transit and at rest, access must be "
        "role-based and logged, and third-party sharing requires a recorded, "
        "purpose-specific consent that the customer can withdraw. Retention "
        "periods must be documented in the privacy notice.",
        "2023-08-01", "2024-06-10",
    ),
    _record(
        "dpdp-2023", "Digital Personal Data Protection Principles", "Government of India",
        "6", "data_privacy",
        "Personal data may be processed only for a lawful purpose with the "
        "consent of the data principal. The data fiduciary must give notice of "
        "the purpose, retain data only as long as necessary and implement "
        "reasonable security safeguards.",
        "2023-08-11", "2024-06-25",
    ),
    _record(
        "sebi-cyber-2020", "Cybersecurity and Data Security Framework", "SEBI",
        "5.2", "data_privacy",
        "Intermediaries must protect client records with encryption, access "
        "controls and audit logs, and must not transmit identity documents over "
        "unencrypted channels such as ordinary email.",
        "2020-12-01", "2024-02-05",
    ),
    _record(
        "rbi-retention-2021", "Retention Limits for Personal Data", "RBI",
        "4.6", "data_privacy",
        "Personal data must not be retained indefinitely. Records must be "
        "deleted or irreversibly anonymised once the purpose of processing is "
        "fulfilled and the statutory retention period has expired.",
        "2021-05-01", "2024-03-01",
    ),
    _record(
        "irdai-data-2022", "Sharing of Policyholder Data", "IRDAI",
        "8.5", "data_privacy",
        "Policyholder data may be shared with marketing partners only after "
        "obtaining specific consent and must be governed by a written data "
        "sharing agreement that records the purpose and retention period.",
        "2022-06-01", "2024-05-02",
    ),
    _record(
        "rbi-records-2017", "Maintenance of Customer Records", "RBI",
        "2.5", "record_keeping",
        "Regulated entities shall maintain records of customer instructions, "
        "advice and transactions for at least five years from the date of the "
        "transaction or the end of the relationship, whichever is later.",
        "2017-08-01", "2022-03-31",
        superseded_by="rbi-records-2022", superseded_date="2022-04-01",
    ),
    _record(
        "rbi-records-2022", "Maintenance of Customer and Advice Records", "RBI",
        "3.1", "record_keeping",
        "Regulated entities must retain records of customer instructions, "
        "advice, suitability assessments and transactions for at least eight "
        "years. Records must be retrievable within the time specified by the "
        "supervisor.",
        "2022-04-01", "2024-06-30",
    ),
    _record(
        "sebi-records-2021", "Record Keeping for Investment Advice", "SEBI",
        "4.4", "record_keeping",
        "Investment advisers must maintain records of the advice given, the "
        "basis for the recommendation and the client's instructions for at "
        "least five years and must produce them on request.",
        "2021-01-01", "2024-01-10",
    ),
    _record(
        "irdai-records-2020", "Policy and Advice Record Retention", "IRDAI",
        "7.2", "record_keeping",
        "Insurers and distributors must retain policy records, proposal "
        "documents and advice for the period prescribed by the regulator and "
        "must not destroy records that are subject to an investigation.",
        "2020-10-01", "2023-11-22",
    ),
    _record(
        "rbi-call-records-2021", "Recording of Telephone Advice", "RBI",
        "5.4", "record_keeping",
        "Where investment or product advice is given over the telephone, a "
        "contemporaneous record or recording must be made and retained in the "
        "client file. Undocumented verbal advice is not an acceptable record.",
        "2021-09-01", "2024-04-12",
    ),
]


def main() -> None:
    ids = [record["doc_id"] for record in RECORDS]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate doc_id in corpus definition")

    os.makedirs(HERE, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        for record in RECORDS:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    categories: dict[str, int] = {}
    for record in RECORDS:
        categories[record["category"]] = categories.get(record["category"], 0) + 1
    superseded = sum(1 for record in RECORDS if record["superseded_by"])
    print(f"wrote {OUTPUT} ({len(RECORDS)} regulations, {superseded} superseded)")
    for category in sorted(categories):
        print(f"  {category:28s} {categories[category]}")


if __name__ == "__main__":
    main()
