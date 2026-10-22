import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_p047_r6_selected as B


class Gate(unittest.TestCase):
    def fixture(self):
        return {'complete':True,'own_Gold_fit_select':0,'selected_variant':'crop320','mode':'four_choices_exact_common_cohort',
                'policy_SHA':B.O.C.POLICY_SHA,'original_policy_SHA':B.O.E.B.POLICY_SHA,
                'counts':{'0':809,'1':855},'paired_successes':{'0':809,'1':855},
                'fixed10_deltas':{name:{'0':.001,'1':.002} for name in B.T.CHOICES},
                'uncropped_fluid_exact_on_all_pairs':True,'crop_combined_replay_variants_evaluated':False}

    def test_recomputed_minimax_both_banks_not_recorded_choice(self):
        r=self.fixture()
        with self.assertRaises(ValueError):
            B.selected(r)
        r['fixed10_deltas']['crop320']={'0':.003,'1':.004}
        self.assertEqual(B.selected(r),'crop320')

    def test_no_winner_pending_gold_mismatch_or_insufficient_coverage_refused(self):
        original=self.fixture()
        original['selected_variant']='tta_off'
        for key,value in [('complete',False),('own_Gold_fit_select',1),('selected_variant',None),
                          ('policy_SHA','x'),('paired_successes',{'0':700,'1':855}),
                          ('crop_combined_replay_variants_evaluated',True)]:
            r=copy.deepcopy(original)
            r[key]=value
            with self.assertRaises(ValueError):
                B.selected(r)

    def test_declared_original_three_fallback_only(self):
        original={'complete':True,'own_Gold_fit_select':0,'policy_SHA':B.O.E.B.POLICY_SHA,
                  'fixed10_deltas':{name:{'0':.001,'1':.002} for name in B.O.E.VARIANTS},
                  'counts':{'0':809,'1':855},'successes':{'0':809,'1':855}}
        r={'complete':True,'own_Gold_fit_select':0,'mode':'original_three_only_crop_engineering_HOLD',
           'selected_variant':'tta_off','original_result':original}
        self.assertEqual(B.selected(r),'tta_off')
        r['selected_variant']='crop320'
        with self.assertRaises(ValueError):
            B.selected(r)


if __name__=='__main__':
    unittest.main()
