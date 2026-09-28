"""Reproducible, retrospective SBA research experiment; not an underwriting system."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
import os
if os.environ.get('CREDIT_RISK_EXTRA_PACKAGES'): sys.path.insert(0, os.environ['CREDIT_RISK_EXTRA_PACKAGES'])
import hashlib, json, platform
import numpy as np
import pandas as pd
import sklearn, xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT / 'results/sba'
OUT.mkdir(exist_ok=True, parents=True)
SOURCE = ROOT / 'data/raw/sba/loans.csv'
ASOF = pd.Timestamp('2026-06-30')
RAW_URL = 'https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY2020_Present_asof_260630.csv'
NUM = ['LogApprovalAmount', 'GuaranteeShare', 'LogJobsSupported']
CAT = ['BusinessAge', 'BusinessType', 'IndustrySector', 'RevolverStatus', 'ProcessingMethod']
FEATURES = NUM + CAT

def prepare_features(d):
    out = pd.DataFrame(index=d.index)
    amount = pd.to_numeric(d.GrossApproval, errors='coerce')
    jobs = pd.to_numeric(d.JobsSupported, errors='coerce')
    out['LogApprovalAmount'] = np.log1p(amount.where(amount >= 0))
    out['GuaranteeShare'] = pd.to_numeric(d.SBAGuaranteedApproval, errors='coerce') / amount.where(amount > 0)
    out['TermInMonths'] = pd.to_numeric(d.TermInMonths, errors='coerce').where(lambda v: v > 0)
    out['LogJobsSupported'] = np.log1p(jobs.where(jobs >= 0))
    out['IndustrySector'] = d.NaicsCode.astype('string').str.replace(r'\.0$', '', regex=True).str[:2]
    for c in CAT:
        if c != 'IndustrySector':
            out[c] = d[c]
        out[c] = out[c].fillna('Unknown').astype(str).str.strip()
    return out[FEATURES].replace([np.inf, -np.inf], np.nan)

def metrics(y, p):
    k = max(1, int(np.ceil(len(y) * .1)))
    top = np.argsort(-p, kind='stable')[:k]
    return dict(roc_auc=float(roc_auc_score(y,p)), average_precision=float(average_precision_score(y,p)),
                brier=float(brier_score_loss(y,p)), log_loss=float(log_loss(y,p)),
                mean_prediction=float(np.mean(p)), observed_rate=float(np.mean(y)),
                top_decile_precision=float(y[top].mean()), top_decile_recall=float(y[top].sum()/y.sum()),
                top_decile_lift=float(y[top].mean()/y.mean()))

def main():
    d = pd.read_csv(SOURCE, low_memory=False)
    assert set(d.AsOfDate.dropna()) == {'2026-06-30'}
    for c in ['ApprovalDate','FirstDisbursementDate','ChargeOffDate','PaidInFullDate']:
        d[c] = pd.to_datetime(d[c], errors='coerce')
    d['horizon_end'] = d.ApprovalDate + pd.DateOffset(months=24)
    status = d.LoanStatus.fillna('').str.replace(' ', '')
    bad_charge_dates = d.ChargeOffDate.notna() & ((d.ChargeOffDate < d.ApprovalDate) | (d.ChargeOffDate > ASOF))
    missing_charge_dates = status.eq('CHGOFF') & d.ChargeOffDate.isna()
    bad_disbursement = d.FirstDisbursementDate.notna() & ((d.FirstDisbursementDate < d.ApprovalDate) | (d.FirstDisbursementDate > ASOF))
    # EXEMPT means status withheld, not verified current/delinquency-free. The target is recorded charge-off only.
    eligible = (d.FirstDisbursementDate.notna() & status.isin(['PIF','CHGOFF','EXEMPT']) &
                (d.horizon_end <= ASOF) & ~bad_charge_dates & ~missing_charge_dates & ~bad_disbursement)
    d['target'] = (d.ChargeOffDate.notna() & (d.ChargeOffDate <= d.horizon_end) & (d.ChargeOffDate >= d.ApprovalDate)).astype(int)
    train_mask = eligible & d.ApprovalDate.between('2019-10-01','2021-06-30')
    test_mask = eligible & d.ApprovalDate.between('2023-07-01','2024-06-30')
    # Remove repeated borrower+address matches from the test cohort; use only in memory, never as predictors.
    identity = d[['BorrName','BorrStreet','BorrZip']].fillna('').astype(str).apply(lambda s:s.str.upper().str.strip())
    key = pd.util.hash_pandas_object(identity, index=False)
    overlap = test_mask & key.isin(set(key[train_mask]))
    test_mask &= ~overlap
    train, test = d.loc[train_mask].copy(), d.loc[test_mask].copy()
    assert train.horizon_end.max() < test.ApprovalDate.min(), 'Outcome maturity overlap'
    assert test.horizon_end.max() <= ASOF
    assert len(train) > 1000 and len(test) > 1000
    Xtrain, Xtest = prepare_features(train), prepare_features(test)
    ytrain, ytest = train.target.to_numpy(), test.target.to_numpy()
    assert ytrain.sum() > 30 and ytest.sum() > 30
    assert not set(FEATURES) & {'LoanStatus','ChargeOffDate','GrossChargeOffAmount','PaidInFullDate','BorrState','TermInMonths'}
    pre = ColumnTransformer([
        ('num', Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),NUM),
        ('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=50,sparse_output=False),CAT)
    ])
    A = pre.fit_transform(Xtrain)
    B = pre.transform(Xtest)
    models = {
        'Logistic regression':LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs',random_state=42),
        'Boosted trees':xgb.XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.04,subsample=.8,
            colsample_bytree=.8,min_child_weight=10,reg_lambda=10,objective='binary:logistic',eval_metric='logloss',
            tree_method='hist',n_jobs=4,random_state=42)
    }
    predictions = {}
    results = {}
    for name,model in models.items():
        print('Training',name,flush=True)
        model.fit(A,ytrain)
        p=model.predict_proba(B)[:,1]
        predictions[name]=p
        results[name]=metrics(ytest,p)
        joblib.dump({'preprocessor':pre,'model':model,'features':FEATURES,'target':'recorded charge-off within 24 months of approval','research_only':True},OUT/(name.lower().replace(' ','_')+'.joblib'))
    base = np.full(len(test),ytrain.mean())
    results['Training prevalence baseline'] = dict(brier=float(brier_score_loss(ytest,base)),log_loss=float(log_loss(ytest,base)),mean_prediction=float(base[0]),observed_rate=float(ytest.mean()))
    # Paired loan-level bootstrap measures sample uncertainty, not lender clustering or future regime uncertainty.
    rng=np.random.default_rng(42)
    boots={k:[] for k in models}
    differences=[]
    for _ in range(200):
        idx=rng.integers(0,len(test),len(test))
        if np.unique(ytest[idx]).size < 2: continue
        for k,p in predictions.items(): boots[k].append(roc_auc_score(ytest[idx],p[idx]))
        differences.append(boots['Boosted trees'][-1]-boots['Logistic regression'][-1])
    for k in models:
        results[k]['roc_auc_ci95']=[float(v) for v in np.quantile(boots[k],[.025,.975])]
    # Exact tree path-dependent Shapley contributions in raw log-odds, not causal effects or adverse-action reasons.
    booster=models['Boosted trees'].get_booster()
    ix=rng.choice(len(test),min(4000,len(test)),replace=False)
    contributions=booster.predict(xgb.DMatrix(B[ix]),pred_contribs=True)
    margin=booster.predict(xgb.DMatrix(B[ix]),output_margin=True)
    error=float(np.abs(contributions.sum(axis=1)-margin).max())
    assert error < 1e-3
    names=pre.get_feature_names_out()
    mapping=[]
    for name in names:
        bare=name.split('__',1)[1]
        if name.startswith('num__'):
            mapping.append(bare.replace('missingindicator_',''))
        else:
            mapping.append(next(c for c in CAT if bare.startswith(c+'_')))
    grouped=np.column_stack([contributions[:,:-1][:,np.array(mapping)==c].sum(axis=1) for c in FEATURES])
    importance=sorted([{'feature':c,'mean_absolute_log_odds_contribution':float(np.abs(grouped[:,j]).mean())} for j,c in enumerate(FEATURES)],key=lambda r:-r['mean_absolute_log_odds_contribution'])
    missing={c:float(Xtrain[c].isna().mean()) for c in NUM}
    def cohort(frame):
        return dict(rows=len(frame),events=int(frame.target.sum()),rate=float(frame.target.mean()),approval_start=str(frame.ApprovalDate.min().date()),approval_end=str(frame.ApprovalDate.max().date()))
    summary=dict(source_url=RAW_URL,source_asof=str(ASOF.date()),source_rows=len(d),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        target='Recorded SBA charge-off within 24 months of approval; not all defaults or delinquency',train=cohort(train),test=cohort(test),
        exclusions=dict(missing_chargeoff_date=int(missing_charge_dates.sum()),invalid_chargeoff_date=int(bad_charge_dates.sum()),invalid_disbursement_date=int(bad_disbursement.sum()),test_borrower_overlap=int(overlap.sum())),
        loan_status_counts=d.LoanStatus.value_counts().to_dict(),features=FEATURES,missing_numeric_training=missing,metrics=results,
        auc_difference_ci95=[float(v) for v in np.quantile(differences,[.025,.975])],shap_sample=len(ix),shap_additivity_max_error=error,
        shap_importance=importance,versions=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,sklearn=sklearn.__version__,xgboost=xgb.__version__))
    (OUT/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Aggregate state diagnostics are descriptive, not a protected-class fair-lending audit.
    diagnostics=[]
    for state,positions in test.reset_index(drop=True).groupby('BorrState').indices.items():
        if len(positions)<200: continue
        yy=ytest[positions]; pp=predictions['Boosted trees'][positions]
        diagnostics.append(dict(state=state,n=len(positions),events=int(yy.sum()),observed_rate=float(yy.mean()),mean_prediction=float(pp.mean())))
    (OUT/'state_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    # No borrower identifiers, scores or row-level records are exported in this report.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    colors=['#375b8c','#d36b35']
    for (name,p),color in zip(predictions.items(),colors):
        fpr,tpr,_=roc_curve(ytest,p)
        axes[0].plot(fpr,tpr,label=f'{name}: {results[name]["roc_auc"]:.3f}',color=color)
        precision,recall,_=precision_recall_curve(ytest,p)
        axes[1].plot(recall,precision,label=f'{name}: {results[name]["average_precision"]:.3f}',color=color)
    axes[0].plot([0,1],[0,1],'--',color='#bbbbbb')
    axes[0].set(xlabel='False-positive rate',ylabel='Recall',title='Ranking performance (ROC-AUC)')
    axes[1].axhline(ytest.mean(),color='#999999',linestyle='--',label=f'Event prevalence: {ytest.mean():.2%}')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Rare-event detection (average precision)',ylim=(0,1))
    for ax in axes[:2]:ax.legend(fontsize=8,loc='best')
    top=importance[:7][::-1]
    axes[2].barh([r['feature'] for r in top],[r['mean_absolute_log_odds_contribution'] for r in top],color='#375b8c')
    axes[2].set(xlabel='Mean absolute contribution in log-odds',title='Boosted-tree explanation drivers')
    fig.suptitle('SBA 7(a) research | 24-month recorded charge-off | Later-cohort test',fontsize=14,fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/'experiment.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
    lines=['# US credit-data research and first experiment','',
        'Research prototype only. This analysis does not approve or reject applications and does not establish regulatory compliance.','',
        '## Dataset selection','',
        '| Dataset | Recent availability | Useful for | Access / limitation |','|---|---|---|---|',
        '| [SBA 7(a)/504](https://data.sba.gov/dataset/7a-504-foia) | June 30, 2026 snapshot | Small-business recorded charge-offs | Public download; approved loans, limited underwriting features |',
        '| [HMDA](https://www.consumerfinance.gov/about-us/newsroom/2025-hmda-data-on-mortgage-lending-now-available/) | 2025 data released March 31, 2026 | Application decisions and fair-lending research | Public; does not supply subsequent default labels |',
        '| [Freddie Mac](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset) | Performance through March 31, 2026 | Mortgage delinquency and loss modeling | Registration and terms apply |',
        '| [Fannie Mae](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data) | July 31, 2026 release; Q1 2026 performance | Mortgage performance modeling | Registration; commercial use restrictions |','',
        'The release date is distinct from origination date. This uses a recent snapshot but deliberately older training loans so their outcomes mature before testing on newer originations.','',
        '## Experiment design','',
        f'- Source: {len(d):,} SBA 7(a) records; snapshot {ASOF.date()}.',
        f'- Training: {len(train):,} funded loans approved {train.ApprovalDate.min().date()} through {train.ApprovalDate.max().date()}; {int(ytrain.sum()):,} events ({ytrain.mean():.3%}).',
        f'- Test: {len(test):,} funded loans approved {test.ApprovalDate.min().date()} through {test.ApprovalDate.max().date()}; {int(ytest.sum()):,} events ({ytest.mean():.3%}).',
        '- Target: a recorded SBA charge-off within 24 calendar months after approval. This is not lifetime default probability, delinquency probability or loss given default.',
        '- Training outcome windows end before the first test approval. All test outcome windows end by the snapshot date. No hyperparameter selection or calibration uses test outcomes.',
        '- Fixed model settings; no oversampling or class weighting. Missing numeric values are imputed from training data; categorical encoding is fitted on training data only.',
        f'- Removed {int(overlap.sum()):,} test loans with matching borrower name, street and ZIP in training. Matching is approximate; it cannot guarantee entity independence.',
        '- Cancelled, undisbursed, missing-disbursement and contradictory-date records are excluded. EXEMPT is not treated as proof of good standing; it means no disclosed terminal status under the dictionary.',
        '- Loans paid off early count as no charge-off by the horizon. Thus the endpoint includes prepayment as a competing outcome.',
        '- Features: log approved amount, guarantee share, log reported jobs, business age, business type, industry sector, revolving indicator and SBA processing method.',
        '- Names, addresses, ZIPs, state, lender identifiers, interest rates, loan term, loan status, charge-off amounts/dates, payoff dates and secondary-market sale indicators are excluded from predictors. Disbursement is an eligibility filter only.',
        '- Approved terms are negotiated lending outcomes. This is risk conditional on an approved loan package, not an applicant-only pre-approval model.','',
        '## Results','',
        '| Model | ROC-AUC (95% bootstrap interval) | Average precision | Brier score | Mean predicted risk | Top 10% recall |',
        '|---|---:|---:|---:|---:|---:|']
    for name in models:
        r=results[name];ci=r['roc_auc_ci95']
        lines.append(f'| {name} | {r["roc_auc"]:.3f} ({ci[0]:.3f}-{ci[1]:.3f}) | {r["average_precision"]:.4f} | {r["brier"]:.5f} | {r["mean_prediction"]:.3%} | {r["top_decile_recall"]:.1%} |')
    lines += ['',f'Test event prevalence / random-ranking average-precision reference: **{ytest.mean():.3%}**. The training-prevalence constant model has Brier score **{results["Training prevalence baseline"]["brier"]:.5f}**.',
        f'Calibration warning: boosted trees predict an average risk of {results["Boosted trees"]["mean_prediction"]:.3%}, versus the observed {ytest.mean():.3%}. Training prevalence was {ytrain.mean():.3%}. Ranking performance does not fix this underestimation. A later calibration cohort and a fresh untouched evaluation are needed before decision use.',
        'Higher ROC-AUC and average precision are better; lower Brier score is better. AUC is ranking, not percentage accuracy. Top-decile recall is an exploratory fixed review-budget diagnostic, not an approval threshold.',
        f'Paired bootstrap 95% interval for boosted-tree minus logistic ROC-AUC: {summary["auc_difference_ci95"]}. Intervals use 200 loan-level resamples and do not account for lender dependence or future economic uncertainty.','',
        '![Experiment results](experiment.png)','',
        '## Explanations','',
        f'Tree path-dependent Shapley contributions were computed on a seeded sample of {len(ix):,} test loans. Contributions add to the raw model margin with maximum absolute error {error:.6g}. One-hot contributions are grouped by original feature before taking mean absolute values.',
        'Largest explanation drivers: '+', '.join(r['feature'] for r in importance[:5])+'.',
        'These are model explanations relative to its tree-path background, not causal effects, validated customer notices or proof of permissible feature use. Correlated features and product selection can alter attribution.','',
        '## Loan-term sensitivity and suspected leakage','',
        'An initial diagnostic run including TermInMonths produced boosted-tree ROC-AUC 0.910 and logistic ROC-AUC 0.735. Loan term dominated the tree explanations. Many early charge-off cases have unusual terms around 100-113 months rather than the common 120 months. The current data dictionary describes term only as length of loan term; it does not establish that this is an immutable origination value.',
        'TermInMonths is therefore excluded from the final experiment. Its initial results are preserved in term_included_diagnostic.json as a warning, not a validated performance claim. The data pattern suggests possible post-origination changes but does not prove their cause. Historical field lineage must be verified with SBA or point-in-time source records.',
        'This exclusion was prompted by inspecting test-set explanations, so the final test is now a development holdout rather than a pristine final evaluation. A new untouched cohort is required before reporting externally validated performance.','',
        '## What the experiment cannot establish','',
        '- The approved/funded-only population does not represent rejected applicants. No reject inference is attempted.',
        '- Charge-off is delayed and narrower than default; hidden EXEMPT statuses may include delinquency. Absence of a recorded charge-off is not proof of repayment ability.',
        '- The current snapshot is a retrospective reconstruction, not historical point-in-time snapshots. Corrections or updated attributes can introduce remaining look-ahead risk.',
        '- Training and test span different economic regimes, including pandemic assistance. Calibration can shift materially; no deployment probability is certified.',
        '- The file lacks verified income, detailed cash flow, debt-service coverage, credit-bureau history and protected-class labels needed for the intended broader validation. State diagnostics are not a fair-lending audit.',
        '- Excluding direct demographic/geographic fields does not remove proxy discrimination. No fair-lending clearance, state-law rules, adverse-action validation or legal review has been completed.',
        '- Feature availability at the intended decision point must be verified with a lender before use. Approval amount and guarantee terms describe the existing loan package.',
        '- Fresh 2025-2026 loans are not labeled negative merely because they have had less than 24 months to develop a charge-off.','',
        '## Next build stage','',
        'Use this as a reproducible small-business research baseline. For a mortgage system, obtain the registered Freddie Mac/Fannie Mae performance data and separately use HMDA for application/fair-lending research; do not assume public files can be reliably joined borrower by borrower. Choose the lender type and states before implementing the regulatory rule layer.',
        '', '## Reproducibility','',
        f'Source file: {RAW_URL}', f'SHA-256: `{summary["sha256"]}`',
        'Software versions: '+json.dumps(summary['versions']),
        'Run `run_experiment.py` with the project dependencies in `packages`. The downloaded raw source and dictionary remain in `data`; results and research model bundles are in `results`. Bundles are local research artifacts; only load trusted joblib files.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'train':summary['train'],'test':summary['test'],'metrics':results,'shap_top':importance[:5]},indent=2),flush=True)

if __name__=='__main__':main()
