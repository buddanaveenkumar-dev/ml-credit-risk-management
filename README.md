# ML in Credit Risk Management

Reproducible US credit-risk research with a Streamlit results viewer. **Research only: not a production credit-decision system or a compliance certification.**

## Results and coverage

| Source | Executed result | Limitation |
|---|---|---|
| SBA 7(a), June 30, 2026 snapshot | Logistic regression and XGBoost; 388,338 source records; 57,086 evaluation loans | Funded small-business loans; recorded charge-off target; snapshot leakage risk |
| HMDA, requested 2023–2025 Maryland subset | Download attempts recorded | Official endpoint returned HTTP 403; no model trained; HMDA has no repayment labels |
| Freddie Mac, release 47 sample | Parsed 1,000 origination records and 1,011 performance rows | Zero matching loan IDs; full data requires access |
| Fannie Mae public sample | Parsed 757 monthly records covering 8 loans | Format sample is too small for credible training |

SBA exploratory evaluation: logistic regression ROC AUC **0.711**, boosted trees **0.741**. Boosted-tree average precision **0.0218**, versus event prevalence **0.00818**. Mean predicted risk **0.00270** materially understates observed events **0.00818**. Do not interpret ranking quality as accurate decision probabilities.

Aggregate metrics, explanations, plots and data-quality checks are in `results/`. Raw data and saved models are excluded from Git.

## Current technology

The published experiment used **local server CPU compute**, and the Streamlit results app runs on a local server.

| Component | Technology used |
|---|---|
| Language and data processing | Python, pandas, NumPy |
| Machine learning | scikit-learn logistic regression; XGBoost boosted trees |
| Model explanations | TreeSHAP contributions calculated by XGBoost |
| Results and charts | Streamlit, Matplotlib |
| Code hosting and automated checks | GitHub, GitHub Actions, Python unittest |
| Storage | Local CSV and model files; aggregate results committed to GitHub |

**Google Cloud, AWS, and NVIDIA GPU / CUDA acceleration were not used.** GitHub hosts the code and runs automated checks; it does not host the interactive app or train the published models. Raw loan files and saved models remain local and are excluded from the repository.

## Build on AWS, GCP or NVIDIA GPUs

The [deployment options guide](docs/deployment-options.md) shows anyone wishing to run this project on AWS, Google Cloud or NVIDIA GPU infrastructure the tools required, setup steps and adaptations they would need to make. This is informational guidance only; we do not develop, provision or deploy these environments as part of this project. Check current costs directly with the chosen provider before starting. The same guide appears under **Cloud & GPU options** in the app. No cloud deployment or GPU benchmark has been performed.

## Run

Use Python 3.12 or later in a virtual environment:

```sh
python -m pip install -r requirements.txt
python scripts/download_public.py --datasets sba freddie fannie hmda
python scripts/run_all.py
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

The download step reports failures independently; successful files remain available. Large SBA downloads can take several minutes. The HMDA endpoint may reject this environment; try the official data browser in your own environment. Re-run a single experiment with `python scripts/run_sba.py`, or only sample checks with `python -m credit_risk.audit`.

The viewer can run from committed aggregate results without downloading data. It does not accept applications or make approval decisions. The optional `CREDIT_RISK_EXTRA_PACKAGES` environment variable supports an existing local Python package directory; normal installations do not need it.

## Experiment design

Training approvals: October 2019–June 2021. Evaluation approvals: July 2023–June 2024. Target: recorded charge-off within 24 months of approval, observed through June 2026. Exclude unfunded/cancelled records, invalid dates, and matching borrowers across cohorts. Transformations are fitted only on training data. Fixed models, paired bootstrap uncertainty and TreeSHAP explanations are reported.

Loan term was removed after investigating suspicious apparent performance. Consequently the current evaluation is a **development holdout**, not pristine final validation. Snapshot variables may still reflect later updates. Outcomes are sparse, and their prevalence shifts substantially. The portfolio does not represent all US borrowers. Protected-class fairness is not established by state-level diagnostics.

## Official sources and access

- [SBA FOIA data](https://data.sba.gov/dataset/7-a-504-foia): downloaded source URL and SHA-256 are recorded in the SBA metrics.
- [HMDA Data Browser API](https://ffiec.cfpb.gov/documentation/api/data-browser): annual application/decision data; not default data.
- [Freddie Mac loan-level data](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset): register and accept current terms for full performance data.
- [Fannie Mae loan performance data](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data): full data requires provider access and acceptance of terms.

The mortgage adapters currently audit **official samples only**. Full-file modeling is not implemented or represented as completed. Once access exists, the next extension must validate schema versions, join loan identifiers, define a fixed delinquency horizon, handle missing months and censoring, and split independent loans chronologically before fitting models.

Recent snapshots do not mean all underlying loans originated recently. Samples include historical loans. Do not combine the four sources into a single training table: populations and outcomes differ.

## Regulatory scope

This repository provides research evidence, not a determination of US legal compliance. ECOA/Regulation B, FCRA, applicable fair-lending and state requirements, privacy, model governance and adverse-action explanations require review for the actual lender, product and use. TreeSHAP output alone does not satisfy those requirements. No live borrower decisions, credit bureau pulls or applicant notifications occur here.

Code is MIT-licensed. Third-party data remains subject to its source terms; the code license grants no rights over data. Do not commit raw loan files or credentials.
