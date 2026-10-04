"""Upload retained numerical training metrics without interrupting the trainer."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from octo.tracking import start_run

ROOT = Path('artifacts/runs/e2e_train_1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'tracking.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((ROOT / 'config.json').read_text())
    run = start_run(name='e2e_train_1', config=config, directory=str(ROOT / 'tracking'))
    failed = True
    try:
        for receipt in config['dataset_artifacts']:
            if receipt['backend'] == 'wandb':
                run.use_artifact(receipt['artifact'], type='dataset')
        (ROOT / 'tracking_run.json').write_text(json.dumps({'url': run.url, 'id': run.id}, indent=2) + '\n')
        print(run.url, flush=True)
        step = 0
        with (ROOT / 'training_metrics.jsonl').open() as handle:
            while True:
                while True:
                    position = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith('\n'):
                        handle.seek(position)
                        break
                    row = json.loads(line)
                    if row['step'] > step:
                        run.log({'train/loss': row['loss'], 'train/step_seconds': row['step_seconds']}, step=row['step'])
                        step = row['step']
                state = json.loads((ROOT / 'status.json').read_text())
                if not args.follow or (ROOT / 'training_complete.json').exists() or state['phase'] in ('failed', 'interrupted', 'complete'):
                    break
                time.sleep(30)
        if (ROOT / 'heldout_metrics.json').exists():
            scores = json.loads((ROOT / 'heldout_metrics.json').read_text())
            run.summary.update({f'{split}/{metric}': value for split, values in scores.items() for metric, value in values.items()})
        run.summary.update({'last_uploaded_training_step': step, 'local_checkpoints': str(ROOT / 'checkpoints'),
                            'checkpoint_policy': 'all retained; every 1000 steps plus initial and final',
                            'base_weights': str(ROOT / 'base_weights')})
        failed = False
    finally:
        run.finish(exit_code=1 if failed else 0)


if __name__ == '__main__':
    main()
