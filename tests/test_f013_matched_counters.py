import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_f013_f009_matched import matched_counters


class Counters(unittest.TestCase):
    def fixture(self):
        return [{'epoch':i, 'optimizer_steps':208-int(i==10), 'optimizer_step_opportunities':208, 'amp_skipped_steps':int(i==10)} for i in range(1,13)]

    def test_exact_matched_known_pattern(self):
        result=matched_counters(self.fixture(),self.fixture())
        self.assertEqual(result['control'],{'optimizer_steps':2495,'optimizer_step_opportunities':2496,'amp_skipped_steps':1})
        self.assertEqual(result['control'],result['candidate'])

    def test_any_different_skip_epoch_count_or_bool_rejected(self):
        original=self.fixture()
        variants=[]
        a=copy.deepcopy(original)
        a[9]['optimizer_steps']=208
        a[9]['amp_skipped_steps']=0
        variants.append(a)
        a=copy.deepcopy(original)
        a[8]['optimizer_steps']=207
        a[8]['amp_skipped_steps']=1
        variants.append(a)
        a=copy.deepcopy(original)
        a[0]['amp_skipped_steps']=False
        variants.append(a)
        for value in variants:
            with self.assertRaises(ValueError):
                matched_counters(original,value)

    def test_missing_extra_or_reordered_epochs_rejected(self):
        original=self.fixture()
        for other in (original[:-1],original+[original[-1]],list(reversed(original))):
            with self.assertRaises(ValueError):
                matched_counters(original,other)


if __name__=='__main__':
    unittest.main()
