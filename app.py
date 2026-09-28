"""Results viewer. Reads aggregate results only; never loads borrower records."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Credit Risk Research', page_icon='📊', layout='wide')

@st.cache_data(ttl=60)
def load_json(relative):
    path = ROOT / relative
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

st.title('ML in Credit Risk Management')
st.caption('US public data • Reproducible experiments • Evidence before decisions')
st.warning('Research prototype. These results do not establish regulatory compliance or support live credit approvals.')
page = st.radio('Explore', ['Overview', 'SBA model results', 'Mortgage data', 'Methods & limitations'], horizontal=True)
metrics = load_json('results/sba/metrics.json')
audits = load_json('results/source_audits.json')

if page == 'Overview':
    st.header('Four sources, an honest view of readiness')
    st.dataframe(pd.DataFrame([
        {'Source':'SBA 7(a)', 'Status':'Two models evaluated' if metrics else 'Not run', 'Task':'24-month recorded charge-off'},
        {'Source':'HMDA', 'Status':'Download blocked (HTTP 403)', 'Task':'Historical lending decisions; no default labels'},
        {'Source':'Freddie Mac', 'Status':'Format sample audited', 'Task':'Full loan-performance access needed'},
        {'Source':'Fannie Mae', 'Status':'Format sample audited', 'Task':'Full loan-performance access needed'},
    ]), hide_index=True, width='stretch')
    if metrics:
        with st.container(horizontal=True):
            st.metric('SBA source records', f"{metrics['source_rows']:,}", border=True)
            st.metric('Evaluation loans', f"{metrics['test']['rows']:,}", border=True)
            st.metric('Boosted-tree ROC AUC', f"{metrics['metrics']['Boosted trees']['roc_auc']:.3f}", border=True)
        st.info('Ranking improves over logistic regression, but predicted event rates are substantially too low. Calibration and a fresh evaluation cohort are required.')

    st.subheader('Current technology')
    st.caption('Technology used for the published experiment and results app on the local server.')
    st.dataframe(pd.DataFrame([
        {'Component': 'Compute', 'Technology used': 'Local server; CPU training'},
        {'Component': 'Language and data processing', 'Technology used': 'Python, pandas, NumPy'},
        {'Component': 'Machine learning', 'Technology used': 'scikit-learn (logistic regression), XGBoost (boosted trees)'},
        {'Component': 'Model explanations', 'Technology used': 'TreeSHAP contributions calculated by XGBoost'},
        {'Component': 'Results and charts', 'Technology used': 'Streamlit, Matplotlib'},
        {'Component': 'Code and automated checks', 'Technology used': 'GitHub, GitHub Actions, Python unittest'},
        {'Component': 'Storage', 'Technology used': 'Local CSV and model files; aggregate results in GitHub'},
    ]), hide_index=True, width='stretch')
    st.caption('Google Cloud and AWS were not used. NVIDIA GPUs / CUDA were not used. The app is hosted on a local server; GitHub hosts the code and runs automated checks.')

elif page == 'SBA model results':
    if not metrics:
        st.info('Run scripts/run_sba.py to generate results.')
        st.stop()
    st.header('SBA 7(a): recorded charge-off within 24 months')
    st.caption(f"Source snapshot: {metrics['source_asof']} · Training: 2019-10 to 2021-06 · Evaluation: 2023-07 to 2024-06")
    frame = pd.DataFrame(metrics['metrics']).T
    cols = ['roc_auc', 'average_precision', 'brier', 'mean_prediction', 'observed_rate', 'top_decile_recall']
    st.dataframe(frame[cols].apply(pd.to_numeric), width='stretch')
    st.caption('ROC AUC: ranking quality; average precision: performance on rare events; Brier: probability error (lower is better). Missing baseline metrics are intentional.')
    st.subheader('Predicted versus observed event rate')
    st.bar_chart(frame[['mean_prediction', 'observed_rate']].apply(pd.to_numeric))
    st.subheader('What influenced the boosted model')
    shap = pd.DataFrame(metrics['shap_importance']).set_index('feature')
    st.bar_chart(shap, horizontal=True)
    st.caption('Mean absolute TreeSHAP contribution in log-odds across 4,000 evaluation rows. Association is not causation or a validated adverse-action reason.')
    if (ROOT/'results/sba/experiment.png').exists():
        st.image(str(ROOT/'results/sba/experiment.png'), width='stretch')
    st.download_button('Download aggregate results', json.dumps(metrics, indent=2), 'sba-metrics.json', 'application/json')

elif page == 'Mortgage data':
    st.header('Source checks and next steps')
    for key in ['hmda', 'freddie', 'fannie']:
        with st.container(border=True):
            record = audits.get(key, {})
            st.subheader({'hmda':'HMDA', 'freddie':'Freddie Mac', 'fannie':'Fannie Mae'}[key])
            st.write(record.get('reason', 'Audit has not run.'))
            st.json({k:v for k,v in record.items() if k not in ['reason', 'attempts']}, expanded=True)
    st.markdown('Full mortgage files require registration and acceptance of provider terms. Keep downloaded data outside GitHub. Format samples cannot support a credible performance claim.')
    st.link_button('Freddie Mac loan-level data', 'https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset')
    st.link_button('Fannie Mae loan performance', 'https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data')
    st.link_button('HMDA data browser', 'https://ffiec.cfpb.gov/data-browser/')

else:
    st.header('What this project establishes')
    st.markdown('''The SBA workflow compares logistic regression and boosted trees using an earlier training cohort and a later evaluation cohort. Preprocessing is fitted on training data. Borrower overlap is removed using a local identity match; raw identifiers are never published.

The target is a **recorded charge-off**, not every default or missed payment. Public snapshot attributes may have changed after origination. Loan term was excluded after a leakage investigation; the evaluation set therefore remains exploratory, not untouched final validation.

Observed event rates shifted sharply between cohorts. Good ranking does not mean calibrated probabilities. The sample covers funded SBA loans and cannot estimate outcomes for rejected applicants.

HMDA describes applications and decisions, not repayment outcomes. Reproducing historical denials could reproduce historical inequities. Mortgage format samples are audited without training models.

Before any real deployment: obtain permitted representative data, define the decision and outcome, reconstruct point-in-time inputs, validate calibration and discrimination on a fresh cohort, investigate group disparities, validate specific reasons for decisions, and obtain legal and independent model review. No model in this project is approved for production underwriting.''')
