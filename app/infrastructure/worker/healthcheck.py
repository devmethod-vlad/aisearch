
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--worker", help="Тип логирования", choices=["main", "queue"], required=True
# )
# args = parser.parse_args()
# if args.worker:
#     if args.worker == 'main':
#         from app.infrastructure.worker.worker import worker
#         result = worker.send_task('main_worker_health_check', queue=f'main-worker-healthcheck')
#     elif args.worker == 'queue':
#         from app.infrastructure.worker.queue_worker import celery_client as queue_worker
#         result = queue_worker.send_task('queue_worker_health_check', queue=f'queue-worker-healthcheck')
#     sys.exit(0 if result.get(timeout=10).get('status') == 'healthy' else 1)
# else:
#     sys.exit(1)

import sys
from app.infrastructure.worker.worker import worker
result = worker.send_task('main_worker_health_check', queue=f'gpu-search')
sys.exit(0 if result.get(timeout=10).get('status') == 'healthy' else 1)