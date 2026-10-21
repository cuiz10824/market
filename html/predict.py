"""One daily inference run; --refresh retrieves recent completed candles first."""
import argparse
import json
from daily.service import DailyService
from daily.agent import WeeklyAgent

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    service=DailyService()
    agent=WeeklyAgent(service,enabled=False)
    if args.refresh:agent.run(force=True)
    service.export_snapshot()
    print(json.dumps({'as_of':sorted({r['as_of'] for r in service.payload['rows']}),
                      'indices':len(service.payload['rows']),'last_refresh':service.last_refresh},ensure_ascii=False,indent=2))
    if args.refresh and agent.state['status'] not in ('ready',):
        raise SystemExit(1)
