
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--search_worker", help="Тип логирования", choices=["main", "queue"], required=True
# )
# args = parser.parse_args()
# if args.search_worker:
#     if args.search_worker == 'main':
#         from app.infrastructure.search_worker.search_worker import search_worker
#         result = search_worker.send_task('main_worker_health_check', queue=f'main-search_worker-healthcheck')
#     elif args.search_worker == 'queue':
#         from app.infrastructure.search_worker.queue_worker import celery_client as queue_worker
#         result = queue_worker.send_task('queue_worker_health_check', queue=f'queue-search_worker-healthcheck')
#     sys.exit(0 if result.get(timeout=10).get('status') == 'healthy' else 1)
# else:
#     sys.exit(1)

import sys
from app.infrastructure.search_worker.worker import worker
result = worker.send_task('main_worker_health_check', queue=f'gpu-search')
sys.exit(0 if result.get(timeout=10).get('status') == 'healthy' else 1)