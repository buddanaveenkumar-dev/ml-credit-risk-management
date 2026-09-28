"""Download official public data only. Restricted full datasets require user registration."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'freddie': ('https://www.freddiemac.com/fmac-resources/research/docs/release-47-sample-files.zip', 'freddie/release47_sample.zip'),
    'freddie_headers': ('https://www.freddiemac.com/fmac-resources/research/docs/file_headers_july_2026.zip', 'freddie/headers.zip'),
    'fannie': ('https://capitalmarkets.fanniemae.com/resources/file/credit-risk/xls/sf-loan-performance-data-sample.csv', 'fannie/sample.csv'),
    'sba': ('https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY2020_Present_asof_260630.csv', 'sba/loans.csv'),
}

def fetch(name, url, relative):
    path = ROOT/'data/raw'/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'CreditRiskResearch/1.0 (public academic research)'})
                with urllib.request.urlopen(req,timeout=180) as response, path.with_suffix(path.suffix+'.partial').open('wb') as f:
                    while chunk:=response.read(1024*1024): f.write(chunk)
                path.with_suffix(path.suffix+'.partial').replace(path)
                break
            except Exception:
                if attempt == 2: raise
                time.sleep(2*(attempt+1))
    return dict(dataset=name,url=url,path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),verified_at=datetime.now(timezone.utc).isoformat())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=['hmda','freddie','fannie'])
    parser.add_argument('--state',default='MD')
    parser.add_argument('--years',nargs='+',type=int,default=[2023,2024,2025])
    args=parser.parse_args()
    tasks=[]
    for name in args.datasets:
        if name=='hmda':
            for year in args.years:
                url=f'https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states={args.state}&years={year}&loan_types=1&loan_purposes=1&lien_statuses=1&actions_taken=1,2,3'
                tasks.append((f'hmda_{args.state}_{year}',url,f'hmda/{args.state}_{year}.csv'))
        else:
            tasks.append((name,*SOURCES[name]))
            if name=='freddie':tasks.append(('freddie_headers',*SOURCES['freddie_headers']))
    manifest=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,*task):task[0] for task in tasks}
        for f in as_completed(futures):
            try:
                record=f.result();manifest.append(record)
                print(record['dataset'],record['bytes'],'bytes',flush=True)
            except Exception as e:
                record={'dataset':futures[f],'error':str(e)};manifest.append(record);print(record,flush=True)
    path=ROOT/'data/download_manifest.json'
    previous=json.loads(path.read_text()) if path.exists() else []
    names={r['dataset'] for r in manifest}
    path.write_text(json.dumps([r for r in previous if r['dataset'] not in names]+manifest,indent=2))
    if any('error' in r for r in manifest):raise SystemExit(1)

if __name__=='__main__':main()
