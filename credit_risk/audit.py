"""Audit official format samples without treating sample rows as a training cohort."""
import io
import json
from pathlib import Path
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def delinquent_90(code):
    value = str(code).strip()
    if value == 'RA':
        return True
    if value in ('99', 'XX', '', 'nan'):
        return None
    if not value.isdigit():
        return None
    return 3 <= int(value) <= 98

def read_pipe(source, minimum_columns):
    frame = pd.read_csv(source, sep='|', header=None, dtype=str, keep_default_na=False)
    if frame.shape[1] < minimum_columns:
        raise ValueError(f'Unsupported schema: {frame.shape[1]} columns; expected at least {minimum_columns}')
    return frame

def audit_freddie(path):
    with zipfile.ZipFile(path) as archive:
        orig_names = [n for n in archive.namelist() if n.endswith('origination_sample_file.txt')]
        perf_names = [n for n in archive.namelist() if n.endswith('performance_sample_file.txt')]
        if len(orig_names) != 1 or len(perf_names) != 1:
            raise ValueError('Expected official origination and performance sample files')
        orig = read_pipe(io.BytesIO(archive.read(orig_names[0])), 24)
        perf = read_pipe(io.BytesIO(archive.read(perf_names[0])), 10)
    joined = set(orig[19]) & set(perf[0])
    return dict(dataset='Freddie Mac', status='format_sample_only', origination_rows=len(orig),
                performance_rows=len(perf), origination_loans=orig[19].nunique(),
                performance_loans=perf[0].nunique(), matched_loans=len(joined),
                origination_columns=orig.shape[1], performance_columns=perf.shape[1],
                models_trained=0, reason='Official format samples are not a representative training cohort. '
                + ('No loan IDs match across the files.' if not joined else 'Obtain licensed full data before modeling.'))

def audit_fannie(path):
    frame = read_pipe(path, 40)
    periods = pd.to_datetime(frame[2], format='%m%Y', errors='coerce')
    return dict(dataset='Fannie Mae', status='format_sample_only', performance_rows=len(frame),
                loans=frame[1].nunique(), columns=frame.shape[1],
                first_report=periods.min().strftime('%Y-%m'), last_report=periods.max().strftime('%Y-%m'),
                models_trained=0, reason='Only a format sample; too few independent loans for credible model evaluation.')

def run():
    result = {}
    for key, function, file in [
        ('freddie', audit_freddie, ROOT/'data/raw/freddie/release47_sample.zip'),
        ('fannie', audit_fannie, ROOT/'data/raw/fannie/sample.csv')]:
        try:
            result[key] = function(file)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            result[key] = dict(status='unavailable', reason=str(exc), models_trained=0)
    manifest = ROOT/'data/download_manifest.json'
    attempts = json.loads(manifest.read_text()) if manifest.exists() else []
    result['hmda'] = dict(status='download_blocked', models_trained=0,
                          reason='Official downloads returned HTTP 403 in this environment. Run the download script or import an official CSV locally.',
                          attempts=[a for a in attempts if a['dataset'].startswith('hmda')])
    output = ROOT/'results/source_audits.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    run()
