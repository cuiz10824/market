import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.three_class import labels, metrics, fit, predict, export, choose, align_probabilities
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'html'))

from inference.three_class_predictor import ThreeClassPredictor
from research.dataset import FEATURES

ROOT = Path(__file__).resolve().parents[2]
P = json.loads((ROOT/'code/training/three_class_protocol.json').read_text())


class ThreeClassTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(labels([-.051,-.05,0,.05,.051]).tolist(),[0,1,1,1,2])
        self.assertEqual(labels([1.05-1, .95-1]).tolist(),[1,1])
        with self.assertRaises(ValueError): labels([float('nan')])

    def test_majority_accuracy_is_not_three_class_skill(self):
        y = [0]*5+[1]*90+[2]*5
        result = metrics(y,np.tile([0,1,0],(100,1)))
        self.assertEqual(result['accuracy'],.9)
        self.assertLess(result['macro_f1'],.32)
        self.assertEqual(result['confusion_matrix'],[[0,5,0],[0,90,0],[0,5,0]])
        self.assertEqual([c['recall'] for c in result['per_class']],[0,1,0])

    def test_selection_uses_validation_f1(self):
        scores = {k:{'macro_f1':.3,'log_loss':1.} for k in P['classification_candidates']}
        scores['logistic_balanced']['macro_f1']=.45
        self.assertEqual(choose([{'scores':scores}],P)[0],'logistic_balanced')
        scores['logistic_balanced']['macro_f1']=.301
        self.assertIn(choose([{'scores':scores}],P)[0],['constant','neutral'])

    def test_missing_classes(self):
        np.testing.assert_allclose(align_probabilities([.2,.7],[0,2]),[[.8,0,.2],[.3,0,.7]])

    def test_serialized_prediction_and_regression_lock(self):
        rng=np.random.default_rng(7)
        for classes in [[-.1,0,.1],[-.1,.1],[0]]:
            rows=[dict(zip(FEATURES,rng.normal(size=len(FEATURES))),target_return=classes[i%len(classes)]) for i in range(120)]
            for kind in P['classification_candidates']:
                fitted=fit(kind,rows,P)
                with tempfile.TemporaryDirectory() as tmp:
                    folder=Path(tmp)
                    reference='code/models/'+P['base_run']+'/holdout/000300/5/model.json'
                    import hashlib
                    bundle={'schema_version':1,'task':'three_class_return','code':'000300','horizon':5,
                            'features':list(FEATURES),'labels':P['labels'],'classifier':export(fitted,folder),
                            'regression_reference':{'bundle':reference,'sha256':hashlib.sha256((ROOT/reference).read_bytes()).hexdigest()}}
                    path=folder/'model.json';path.write_text(json.dumps(bundle))
                    loaded=ThreeClassPredictor(path)
                    outputs=loaded.predict(rows)
                    np.testing.assert_allclose([[o['p_big_down'],o['p_neutral'],o['p_big_up']] for o in outputs],predict(fitted,rows),atol=1e-12)
                    self.assertEqual([o['expected_price_return'] for o in outputs],
                                     [o['expected_price_return'] for o in loaded.regression.predict(rows)])
                    bundle['regression_reference']['sha256']='bad'
                    path.write_text(json.dumps(bundle))
                    with self.assertRaises(ValueError):ThreeClassPredictor(path)


if __name__=='__main__':unittest.main()
