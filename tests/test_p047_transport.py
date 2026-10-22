import ast
import base64
import hashlib
import json
import lzma
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_p047_r6_selected as B


class Transport(unittest.TestCase):
    def test_all_four_hypothetical_winners_preserve_other7_and_inputs(self):
        # Synthetic selection reports are local test fixtures, never real evidence.
        with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts/research',prefix='test-p047-transport-') as temp:
            root=Path(temp)
            original=[B.F.text(c) for c in B.F.read(B.F.P033/'source.ipynb')['cells'] if c['cell_type']=='code']
            oldpayload=B.F.read(B.P041/'save-payload.json')
            for choice in B.T.CHOICES:
                report={'complete':True,'own_Gold_fit_select':0,'selected_variant':choice,'mode':'four_choices_exact_common_cohort',
                        'policy_SHA':B.O.C.POLICY_SHA,'original_policy_SHA':B.O.E.B.POLICY_SHA,
                        'counts':{'0':809,'1':855},'paired_successes':{'0':809,'1':855},
                        'fixed10_deltas':{name:{'0':.003 if name==choice else .001,'1':.004 if name==choice else .002} for name in B.T.CHOICES},
                        'uncropped_fluid_exact_on_all_pairs':True,'crop_combined_replay_variants_evaluated':False}
                result=root/(choice+'-fake-result.json')
                result.write_text(json.dumps(report),encoding='utf8')
                output=root/(choice+'-fake-payload')
                m=B.prepare(output,'rsraki/test-p047-'+choice.replace('_','-'),result)
                p=json.loads((output/'save-payload.json').read_bytes())
                codes=[B.F.text(c) for c in json.loads(p['text'])['cells'] if c['cell_type']=='code']
                tree=ast.parse(codes[1])
                packed=ast.literal_eval(B.F.assignment(codes[1],'_p047_packed')[1].value.args[0])
                raw=lzma.decompress(base64.b85decode(packed))
                science=json.loads(raw)
                self.assertEqual([s for i,s in enumerate(science) if i!=5],[s for i,s in enumerate(original) if i!=5])
                self.assertEqual(science[5],B.T.transform_stage(original[5],choice)[0])
                self.assertEqual({k:v for k,v in p.items() if k not in ('slug','new_title','text')},
                                 {k:v for k,v in oldpayload.items() if k not in ('slug','new_title','text')})
                config=ast.literal_eval(tree.body[-1].value.args[-1])
                self.assertEqual(config['descriptor']['choice'],choice)
                self.assertEqual(hashlib.sha256(config['original_worker'].encode()).hexdigest(),B.T.WORKER_SHA)
                self.assertLess(m['source_bytes'],950000)
                self.assertEqual(p['text'].encode(),(output/'source.ipynb').read_bytes())
                self.assertEqual(m['selection_result']['sha256'],hashlib.sha256(result.read_bytes()).hexdigest())
                for c in codes:
                    ast.parse(c)


if __name__=='__main__':
    unittest.main()
