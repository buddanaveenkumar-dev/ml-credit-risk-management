"""Run the SBA experiment and audit the other three sources; never invent unavailable results."""
from pathlib import Path
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('CREDIT_RISK_EXTRA_PACKAGES'):
    sys.path.insert(0, os.environ['CREDIT_RISK_EXTRA_PACKAGES'])
sys.path.insert(0, str(ROOT))
from credit_risk.audit import run

if __name__ == '__main__':
    run()
    if (ROOT/'data/raw/sba/loans.csv').exists():
        subprocess.run([sys.executable, str(ROOT/'scripts/run_sba.py')], check=True)
    else:
        print('SBA unavailable: run python scripts/download_public.py --datasets sba')
