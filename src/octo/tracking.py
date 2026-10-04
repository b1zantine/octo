"""W&B metrics and bounded artifact publishing, with a local Docker MLflow fallback."""
import os
from pathlib import Path


def start_run(*, project='octo_v1', name=None, mode='online', config=None, directory='outputs'):
    import wandb
    # Read a single credential assignment as data, without executing shell content.
    if mode == 'online' and not os.environ.get('WANDB_API_KEY'):
        env = Path('.env')
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith('WANDB_API_KEY='):
                    os.environ['WANDB_API_KEY'] = line.partition('=')[2].strip().strip('\"\'')
                    break
    Path(directory).mkdir(parents=True, exist_ok=True)
    return wandb.init(project=project, name=name, mode=mode, config=config or {}, dir=directory,
                      settings=wandb.Settings(disable_git=True, save_code=False))


def log_artifact(run, path, *, name, kind, metadata=None, max_bytes=256 * 1024 * 1024):
    """Upload a bounded dataset or prediction checkpoint, never a repository tree."""
    import wandb
    path = Path(path).resolve()
    if not path.is_dir() or path.is_symlink():
        raise ValueError('artifact must be a directory')
    files = [p for p in path.rglob('*') if p.is_file()]
    if any(p.is_symlink() or p.name == '.env' for p in path.rglob('*')):
        raise ValueError('artifact contains a symlink or credential file')
    size = sum(p.stat().st_size for p in files)
    if size > max_bytes:
        raise ValueError(f'artifact exceeds upload budget: {size} > {max_bytes} bytes; use local MLflow')
    artifact = wandb.Artifact(name, type=kind, metadata={**(metadata or {}), 'size_bytes': size})
    artifact.add_dir(str(path))
    logged = run.log_artifact(artifact)
    if run.settings.mode == 'offline':
        return f'{name}:offline'
    logged.wait(timeout=120)
    return logged.qualified_name


def publish(path, *, kind='dataset', name=None, project='octo_v1', dataset_artifact=None,
            max_bytes=256 * 1024 * 1024, backend='auto', benchmark_artifact=None, model_artifact=None):
    """Publish to W&B; on failure retain artifacts locally and start Docker MLflow."""
    import json
    path = Path(path).resolve()
    artifact_root = Path('artifacts').resolve()
    if not path.is_relative_to(artifact_root):
        raise ValueError('managed artifacts must live under the repo artifacts directory')
    metadata = json.loads((path / ('octo.json' if kind == 'model' else 'manifest.json')).read_text())
    name = name or path.name
    import re
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('invalid artifact name')
    run = None
    failure = None
    if backend != 'mlflow':
        try:
            run = start_run(project=project, name=f'publish-{name}', config={'artifact_type': kind}, directory='artifacts/tracking')
            if dataset_artifact:
                run.use_artifact(dataset_artifact, type='dataset')
            if benchmark_artifact:
                run.use_artifact(benchmark_artifact, type='benchmark')
            if model_artifact:
                run.use_artifact(model_artifact, type='model')
            identity = log_artifact(run, path, name=name, kind=kind, metadata=metadata, max_bytes=max_bytes)
            result = {'backend': 'wandb', 'artifact': identity, 'run_url': run.url}
        except Exception as exc:
            failure = type(exc).__name__
            if backend == 'wandb':
                raise
        finally:
            if run:
                run.finish(exit_code=1 if failure else 0)
    if backend == 'mlflow' or failure:
        import subprocess
        import time
        import requests
        import mlflow
        (artifact_root / 'mlflow').mkdir(parents=True, exist_ok=True)
        subprocess.run(['docker', 'compose', '-f', 'compose.mlflow.yaml', 'up', '-d'], check=True)
        uri = 'http://127.0.0.1:5000'
        for _ in range(30):
            try:
                if requests.get(uri + '/health', timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('MLflow health check failed; local artifacts are preserved')
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(project)
        with mlflow.start_run(run_name=f'publish-{name}') as local_run:
            mlflow.set_tags({'artifact_type': kind, 'artifact_name': name, 'wandb_failure': failure or '',
                            'dataset_artifact': dataset_artifact or '', 'benchmark_artifact': benchmark_artifact or '',
                            'model_artifact': model_artifact or ''})
            mlflow.log_dict(metadata, 'provenance.json')
            mlflow.log_artifacts(str(path), artifact_path=name)
            result = {'backend': 'mlflow', 'run_id': local_run.info.run_id, 'tracking_uri': uri}
    receipts = artifact_root / 'tracking' / 'receipts'
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Publish dataset or model artifacts; auto falls back to Docker MLflow.')
    parser.add_argument('path')
    parser.add_argument('--kind', choices=['dataset', 'model', 'benchmark', 'evaluation'], default='dataset')
    parser.add_argument('--name')
    parser.add_argument('--project', default='octo_v1')
    parser.add_argument('--dataset-artifact')
    parser.add_argument('--benchmark-artifact')
    parser.add_argument('--model-artifact')
    parser.add_argument('--backend', choices=['auto', 'wandb', 'mlflow'], default='auto')
    args = parser.parse_args()
    print(json.dumps(publish(args.path, kind=args.kind, name=args.name, project=args.project,
                             dataset_artifact=args.dataset_artifact, backend=args.backend,
                             benchmark_artifact=args.benchmark_artifact, model_artifact=args.model_artifact), indent=2))

if __name__ == '__main__':
    main()
