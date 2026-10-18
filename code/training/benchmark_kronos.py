"""Pinned official Kronos-mini zero-shot diagnostic; never auto-select pretrained weights."""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'html'),str(ROOT/'code/references/kronos')]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def download_weights(cache):
    """Only official HTTPS destinations; never use alternative mirrors after denial."""
    manifests=[]
    for repo in ('NeoQuasar/Kronos-mini','NeoQuasar/Kronos-Tokenizer-2k'):
        folder=cache/repo.split('/')[-1]
        manifest_path=folder/'manifest.json'
        if manifest_path.exists():
            m=json.loads(manifest_path.read_text())
            for name,digest in m['files'].items():
                if sha(folder/name)!=digest:raise ValueError('Cached weight digest mismatch')
            manifests.append(m);continue
        request=urllib.request.Request('https://huggingface.co/api/models/'+repo+'?blobs=true',headers={'User-Agent':'market-research'})
        with urllib.request.urlopen(request,timeout=30) as response:metadata=json.load(response)
        revision=metadata['sha'];files={};folder.mkdir(parents=True,exist_ok=True)
        for name in ('config.json','model.safetensors'):
            url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
            temp=folder/(name+'.tmp')
            with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as out:
                while True:
                    block=response.read(1024*1024)
                    if not block:break
                    out.write(block)
            sibling=next(s for s in metadata['siblings'] if s['rfilename']==name)
            expected=sibling.get('lfs',{}).get('sha256')
            if expected and sha(temp)!=expected:raise ValueError('Official LFS SHA256 differs')
            temp.replace(folder/name);files[name]=sha(folder/name)
        manifest={'repo':repo,'revision':revision,'files':files,
                  'pretraining_cutoff_verified':False,'fetched_at':dt.datetime.now(dt.timezone.utc).isoformat()}
        write(manifest_path,manifest);manifests.append(manifest)
    return manifests


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-only',action='store_true')
    args=parser.parse_args()
    p=json.loads((ROOT/'code/training/enhanced_protocol.json').read_text())
    cache=ROOT/'.model-cache/kronos'
    output=ROOT/'data/reports/kronos_mini'
    vendor=ROOT/'code/references/kronos'
    for name,digest in json.loads((vendor/'manifest.json').read_text())['files'].items():
        if sha(vendor/name)!=digest:raise ValueError('Kronos vendor file changed')
    try:weights=download_weights(cache)
    except Exception as exc:
        write(output/'status.json',{'status':'blocked','operation':'download official pretrained weights',
              'reason':str(exc),'tested':False,'required_host':'huggingface.co',
              'remedy':'Review environment settings and publish the saved network configuration; then rerun this script.',
              'source_commit':json.loads((vendor/'manifest.json').read_text())['commit']})
        raise SystemExit('Kronos benchmark blocked: '+str(exc))
    if args.weights_only:
        print('Official weight cache verified');return
    import numpy as np
    import pandas as pd
    import torch
    from model import Kronos,KronosTokenizer,KronosPredictor
    from research.rich_dataset import load_bars
    from training.train_enhanced import metrics,old_reports
    torch.set_num_threads(2)
    model=Kronos.from_pretrained(str(cache/'Kronos-mini')).eval()
    tokenizer=KronosTokenizer.from_pretrained(str(cache/'Kronos-Tokenizer-2k')).eval()
    predictor=KronosPredictor(model,tokenizer,device='cpu',max_context=p['kronos']['context'])
    bars,calendar=load_bars();codes=p['codes'];context=p['kronos']['context'];horizons=p['horizons']
    _,old,locked=old_reports()
    origins=[i for i,d in enumerate(calendar) if d>='2025-09-01' and i+max(horizons)<len(calendar)]
    origins=origins[::p['kronos']['origin_stride']]
    rows=[]
    for i in origins:
        frames=[];past=[];future=[]
        for code in codes:
            frame=bars[code].set_index('date').reindex(calendar[i-context+1:i+1])
            if frame[['open','high','low','close','volume']].isna().any().any():raise ValueError('Incomplete Kronos input')
            # Unknown actual amount is a zero missing-channel placeholder, not observed turnover.
            frame=frame.copy();frame['amount']=0.
            frames.append(frame);past.append(pd.Series(pd.to_datetime(frame.index)))
            future.append(pd.Series(pd.to_datetime(calendar[i+1:i+max(horizons)+1])))
        seed=p['seed']+i;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        with torch.inference_mode():
            forecast=predictor.predict_batch(frames,past,future,pred_len=max(horizons),sample_count=p['kronos']['samples'],verbose=False)
        for code,frame,out in zip(codes,frames,forecast):
            close=float(frame.close.iloc[-1])
            for h in horizons:
                predicted=float(out.close.iloc[h-1])
                if not np.isfinite(predicted) or predicted<=0:raise ValueError('Invalid Kronos predicted close')
                reference=old[(code,h,calendar[i])]
                rows.append({'code':code,'horizon':h,'date':calendar[i],'label_end_date':calendar[i+h],
                             'expected_return':predicted/close-1,'actual_return':float(reference['actual_return']),
                             'legacy_return':float(reference['expected_return']),'zero_return':0.})
        print('Kronos completed origin '+calendar[i],flush=True)
    if not rows:raise ValueError('No Kronos evaluation origins')
    with (output/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tasks=[]
    for code in codes:
        for h in horizons:
            group=[r for r in rows if r['code']==code and r['horizon']==h]
            actual=[r['actual_return'] for r in group]
            tasks.append({'code':code,'horizon':h,'kronos':metrics(actual,[r['expected_return'] for r in group]),
                          'zero':metrics(actual,[0.]*len(group)),
                          'legacy':metrics(actual,[r['legacy_return'] for r in group])})
    for path,digest in locked.items():
        if sha(ROOT/path)!=digest:raise ValueError('Original artifact changed')
    write(output/'status.json',{'status':'completed','tested':True,'rows':len(rows),'origin_count':len(origins),
          'weights':weights,'tasks':tasks,'diagnostic_only':True,'pretraining_cutoff_verified':False,
          'amount_channel':'zero missing placeholder, not verified amount data',
          'limitation':'Sparse matched origins; unknown pretraining overlap. Not equivalent to full daily independent test.'})
    print('Kronos diagnostic completed')


if __name__=='__main__':run()
