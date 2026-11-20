"""Validate evidence ZIPs, retain attempts and rebuild results next to notebooks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

BASE = Path(__file__).resolve().parents[1] / 'notebooks/kaggle'
PAPER = {
    ('fewrel',5): (76.96,78.86,77.90), ('fewrel',10): (56.92,57.59,57.25),
    ('fewrel',15): (35.54,38.19,36.82), ('wikizsl',5): (71.54,72.39,71.96),
    ('wikizsl',10): (60.51,60.98,60.74), ('wikizsl',15): (34.12,34.38,34.25),
}


def validate(files):
    manifest = json.loads(files['manifest.json'])
    required = {'experiment_config.json', 'status.json', 'provenance.json',
                'attributes.npz', 'relation_descriptions.json'}
    if not required <= set(manifest):
        raise ValueError('Evidence manifest is missing required files.')
    for name, expected in manifest.items():
        if name not in files or hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError(f'Bad evidence checksum: {name}')
    cfg = json.loads(files['experiment_config.json'])
    if cfg['smoke']:
        raise ValueError('Smoke tests cannot be imported as official results.')
    scientific = {k: v for k, v in cfg.items() if k not in {
        'config_hash', 'seed', 'result_dir', 'attribute_path', 'data_path', 'bert_path',
        'checkpoint_every', 'log_every', 'deadline_unix'}}
    if hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest() != cfg['config_hash']:
        raise ValueError('Config hash does not match configuration.')
    seed = cfg['seed']
    if seed not in range(300, 305) or cfg['m'] not in [5, 10, 15] or cfg['epochs'] != 10:
        raise ValueError('Unexpected experiment configuration.')
    status = json.loads(files['status.json'])
    if status['config_hash'] != cfg['config_hash'] or status['seed'] != seed:
        raise ValueError('Status/config mismatch.')
    if status['status'] == 'complete':
        if not {f'run_seed_{seed}.json', f'history_seed_{seed}.json', f'train_seed_{seed}.log'} <= set(manifest):
            raise ValueError('Completed evidence is missing result, history or log.')
        result = json.loads(files[f'run_seed_{seed}.json'])
        if not result['complete'] or result['config_hash'] != cfg['config_hash'] or result['seed'] != seed or result['m'] != cfg['m']:
            raise ValueError('Result/config mismatch.')
        if [r['epoch'] for r in result['history']] != list(range(1, 11)):
            raise ValueError('Completed results require all ten epochs.')
        if len(result['test_relations']) != cfg['m'] or set(result['test_relations']) & set(result['train_relations']):
            raise ValueError('Invalid seen/unseen split.')
        if result['final'] != result['history'][-1] or result['best'] != max(result['history'], key=lambda r:r['f1']):
            raise ValueError('Best/final differs from history.')
        if json.loads(files[f'history_seed_{seed}.json']) != result['history']:
            raise ValueError('History file differs from completed result.')
        for row in result['history']:
            if any(not isinstance(row.get(k), (int,float)) or not 0 <= row[k] <= 1
                   for k in ['precision','recall','f1']):
                raise ValueError('Metric must be finite and in [0,1].')
    elif status['status'] != 'paused':
        raise ValueError('Unknown run status.')
    return cfg, status


def import_zip(path, label, kaggle_url, base=BASE):
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
        raise ValueError('Attempt label must contain only letters, digits, underscore or dash.')
    if not kaggle_url.startswith('https://www.kaggle.com/code/'):
        raise ValueError('Supply the exact Kaggle version URL as provenance.')
    with zipfile.ZipFile(path) as z:
        files = {}
        roots = set()
        for info in z.infolist():
            parts = info.filename.split('/')
            if len(parts) != 2 or not all(parts) or any(x in ['.', '..'] for x in parts):
                raise ValueError('Unexpected ZIP path; expected run/file only.')
            if parts[1] in files or info.file_size > 30_000_000:
                raise ValueError('Duplicate or oversized evidence file.')
            if Path(parts[1]).suffix not in ['.json', '.log', '.npz']:
                raise ValueError('Unexpected evidence file type.')
            roots.add(parts[0])
            files[parts[1]] = z.read(info)
    if len(roots) != 1:
        raise ValueError('One ZIP must contain exactly one run.')
    cfg, status = validate(files)
    name = f"{cfg['dataset']}_m{cfg['m']}_seed{cfg['seed']}"
    destination = base/'runs'/name/label
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (destination/name).write_bytes(content)
    (destination/'import_record.json').write_text(json.dumps({'kaggle_version_url': kaggle_url,
        'evidence_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    return destination


def rebuild(base=BASE):
    rows, groups = [], {}
    for config in sorted((base/'runs').glob('*/*/experiment_config.json')):
        directory = config.parent
        files = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        cfg, status = validate(files)
        item = dict(dataset=cfg['dataset'], m=cfg['m'], seed=cfg['seed'], attempt=directory.name,
                    status=status['status'], config_hash=cfg['config_hash'], evidence=str(directory.relative_to(base)))
        result = None
        if status['status'] == 'complete':
            result = json.loads(files[f"run_seed_{cfg['seed']}.json"])
        for mode in ['best', 'final']:
            for metric in ['precision', 'recall', 'f1']:
                item[f'{mode}_{metric}'] = result[mode][metric] if result else ''
        rows.append(item)
        if result:
            groups.setdefault((cfg['dataset'], cfg['m'], cfg['config_hash']), []).append(result)
    fields = ['dataset','m','seed','attempt','status','config_hash','evidence'] + [f'{a}_{b}' for a in ['best','final'] for b in ['precision','recall','f1']]
    base.mkdir(parents=True, exist_ok=True)
    with open(base/'results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)
    lines = ['# Kết quả thí nghiệm', '', 'Đơn vị results.csv: 0–1. Bên dưới: phần trăm; std mẫu (ddof=1).',
             'Best chọn epoch bằng unseen test F1; final là epoch 10.',
             'Mỗi attempt được giữ riêng. Không tự chọn attempt có điểm tốt hơn.', '']
    if not rows:
        lines += ['Chưa nhập bằng chứng của bộ chạy mới.', '']
    for (dataset, m, cfg_hash), results in groups.items():
        seeds = [r['seed'] for r in results]
        lines += [f'## {dataset}, m={m}, config {cfg_hash[:12]}', '']
        if len(seeds) != 5 or set(seeds) != set(range(300,305)):
            lines += [f'Chưa tổng hợp: cần đúng một kết quả cho mỗi seed 300–304; hiện có {seeds}.', '']
            continue
        for mode in ['best','final']:
            for metric in ['precision','recall','f1','standard_macro_precision','standard_macro_recall','standard_macro_f1']:
                if not all(metric in r[mode] for r in results):
                    continue
                values = [100*r[mode][metric] for r in results]
                mean = statistics.mean(values)
                delta = ''
                if metric in ['precision','recall','f1']:
                    target = PAPER[(dataset,m)][['precision','recall','f1'].index(metric)]
                    delta = f'; paper {target:.2f}; delta {mean-target:+.2f} pp'
                lines.append(f'- {mode} {metric}: {mean:.2f} ± {statistics.stdev(values):.2f}{delta}')
        lines.append('')
    (base/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--zip', type=Path)
    p.add_argument('--attempt', default='original')
    p.add_argument('--kaggle-version-url', default='')
    args = p.parse_args()
    if args.zip:
        print(import_zip(args.zip, args.attempt, args.kaggle_version_url))
    rebuild()
