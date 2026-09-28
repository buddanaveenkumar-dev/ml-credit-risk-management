import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from credit_risk.audit import delinquent_90, read_pipe, audit_freddie

class DataQualityTests(unittest.TestCase):
    def test_unknown_status_is_not_default(self):
        for code in ['99', 'XX', '', 'not-a-code']:
            self.assertIsNone(delinquent_90(code))
        for code in ['03', '12', 'RA']:
            self.assertTrue(delinquent_90(code))
        for code in ['00', '01', '02']:
            self.assertFalse(delinquent_90(code))

    def test_schema_rejected(self):
        with self.assertRaises(ValueError):
            read_pipe(io.StringIO('1|2|3\n'), 40)

    def test_join_requires_actual_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.zip'
            orig = ['0']*31
            orig[19] = 'loan-A'
            perf = ['0']*35
            perf[0] = 'loan-B'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('origination_sample_file.txt', '|'.join(orig))
                archive.writestr('performance_sample_file.txt', '|'.join(perf))
            result = audit_freddie(path)
            self.assertEqual(result['matched_loans'], 0)
            self.assertEqual(result['models_trained'], 0)

if __name__ == '__main__':
    unittest.main()
