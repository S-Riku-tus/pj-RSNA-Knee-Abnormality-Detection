import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import p047_r6_runtime as R


class Runtime(unittest.TestCase):
    def fixture(self, work):
        cells=['# artificial_%d'%i for i in range(8)]
        inventory=[{'bytes':len(s.encode()),'sha256':hashlib.sha256(s.encode()).hexdigest()} for s in cells]
        source="def parent_observation(ns,work,root):\n return [{'StudyInstanceUID':'fake'}],[False]\n"
        config={'descriptor':{'choice':'tta_off'},'selection_result_SHA':'fake-test-only',
                'observer_source':source,'observer_SHA':hashlib.sha256(source.encode()).hexdigest()}
        calls=[]
        shell=SimpleNamespace(user_ns={},run_cell=lambda text,**kw:(calls.append(text) or SimpleNamespace(error_before_exec=None,error_in_exec=None)))
        Path(work,'submission.csv').write_text('fake preserved csv\n',encoding='utf8')
        Path(work,'submission_receipt.json').write_text('{}',encoding='utf8')
        return cells,inventory,config,calls,shell

    def test_actual_QA_must_finish_between_asset_stage_and_R6_test_inference(self):
        with tempfile.TemporaryDirectory() as work,contextlib.redirect_stdout(io.StringIO()):
            cells,inventory,config,calls,shell=self.fixture(work)
            def qa(*args):
                self.assertEqual(calls,cells[:6])
                return {'selected':0.,'reference':0.}
            with patch.object(R,'artificial_qa',qa):
                result=R.execute(cells,shell,inventory,config,work=work)
            self.assertTrue(result['complete'])
            self.assertEqual(calls,cells)
            self.assertEqual(Path(work,'P047/final-submission.csv').read_bytes(),Path(work,'submission.csv').read_bytes())

    def test_QA_failure_cannot_execute_R6_test_or_claim_complete(self):
        with tempfile.TemporaryDirectory() as work,contextlib.redirect_stdout(io.StringIO()):
            cells,inventory,config,calls,shell=self.fixture(work)
            with patch.object(R,'artificial_qa',side_effect=ValueError('artificial failure')),self.assertRaises(ValueError):
                R.execute(cells,shell,inventory,config,work=work)
            self.assertEqual(calls,cells[:6])
            record=json.loads(Path(work,'P047/execution.json').read_bytes())
            self.assertFalse(record['complete'])
            self.assertFalse(Path(work,'P047/final-submission.csv').exists())

    def test_source_tamper_refused_before_any_execution_or_directory(self):
        with tempfile.TemporaryDirectory() as work:
            cells,inventory,config,calls,shell=self.fixture(work)
            cells[0]+='tampered'
            with self.assertRaises(ValueError):
                R.execute(cells,shell,inventory,config,work=work)
            self.assertFalse(calls)
            self.assertFalse(Path(work,'P047').exists())


if __name__=='__main__':
    unittest.main()
