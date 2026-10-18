"""One daily inference run; --refresh retrieves recent completed candles first."""
import argparse
import json
from daily.service import DailyService

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    service=DailyService()
    if args.refresh:service.refresh()
    service.export_snapshot()
    print(json.dumps({'as_of':sorted({r['as_of'] for r in service.payload['rows']}),
                      'indices':len(service.payload['rows']),'last_refresh':service.last_refresh},ensure_ascii=False,indent=2))
