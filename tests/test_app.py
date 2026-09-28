import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class ResultsViewerTests(unittest.TestCase):
    def test_all_views_render(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        for page in ['SBA model results', 'Mortgage data', 'Methods & limitations']:
            app.radio[0].set_value(page).run(timeout=30)
            self.assertEqual(len(app.exception), 0, page)

if __name__ == '__main__':
    unittest.main()
