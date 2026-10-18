import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'html'))
from daily.rich_features import build_frames,feature_names
from daily.enhanced_predictor import EnhancedPredictor
from research.rich_dataset import samples
from training.enhanced_models import fit,predict,export

CODES=['000510','000300','000928','000933','000932','000930','000934','000935']


def history():
    calendar=pd.bdate_range('2010-01-04',periods=650).strftime('%Y-%m-%d').tolist()
    bars={}
    for j,code in enumerate(CODES):
        x=np.arange(len(calendar));close=100*np.exp(.0002*x+.03*np.sin(x*.14+j))
        bars[code]=pd.DataFrame({'date':calendar,'open':close*.998,'high':close*1.01,'low':close*.99,
                                'close':close,'volume':10000+100*np.sin(x*.3+j)+x})
    return bars,calendar


class EnhancedTests(unittest.TestCase):
    def test_cross_index_and_volume_features_never_use_future_observations(self):
        bars,calendar=history();original=build_frames(bars,CODES,calendar)
        changed={c:f.copy() for c,f in bars.items()}
        for frame in changed.values():
            mask=frame.date>calendar[400]
            frame.loc[mask,['open','high','low','close','volume']]*=2
        altered=build_frames(changed,CODES,calendar)
        for code in CODES:
            pd.testing.assert_frame_equal(original[code].loc[:calendar[400]],altered[code].loc[:calendar[400]])
        self.assertEqual(len(feature_names(CODES)),80)

    def test_labels_follow_observed_calendar_and_cannot_skip_gaps(self):
        bars,calendar=history();missing=calendar[430]
        bars['000300']=bars['000300'][bars['000300'].date!=missing]
        result,_=samples(bars,CODES,calendar)
        positions={d:i for i,d in enumerate(calendar)}
        for h,frame in result.items():
            for row in frame.itertuples():
                self.assertEqual(positions[row.label_end_date]-positions[row.date],h)
                if row.code=='000300':self.assertFalse(row.date<=missing<=row.label_end_date)
        feature=build_frames(bars,CODES,calendar)['000300'].loc[missing]
        self.assertTrue(feature.isna().any())

    def test_all_four_native_models_reload_and_reject_tampered_weights(self):
        protocol=json.loads((ROOT/'code/training/enhanced_protocol.json').read_text())
        protocol['lightgbm']['n_estimators']=15;protocol['catboost']['iterations']=15
        names=feature_names(CODES);rng=np.random.default_rng(71)
        data=pd.DataFrame(rng.normal(size=(350,len(names))),columns=names)
        data['target_return']=data.iloc[:,0]*.01+rng.normal(scale=.005,size=len(data))
        for kind in protocol['candidates']:
            model=fit(kind,data,names,protocol)
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)
                bundle={'schema_version':1,'task':'enriched_return','horizon':1,'features':list(names),
                        'return':export(model,kind,folder)}
                path=folder/'model.json';path.write_text(json.dumps(bundle))
                loaded=EnhancedPredictor(path)
                np.testing.assert_allclose(loaded.predict(data.to_dict('records')),predict(model,kind,data,names),atol=1e-12,rtol=1e-10)
                native=folder/bundle['return']['file'];native.write_bytes(native.read_bytes()+b'corrupt')
                with self.assertRaises(ValueError):EnhancedPredictor(path)

    def test_kronos_vendored_source_digests(self):
        folder=ROOT/'code/references/kronos'
        manifest=json.loads((folder/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
