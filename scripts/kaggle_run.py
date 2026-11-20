"""Prepare, run and export one reproducible experiment on Kaggle."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from normalize_data import preprocess
from dataset_source import prepare_snapshot

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'fewrel': ('1QY-5R2zqLPnT5DDF5phTbSvEeBkQRh7A', 80, 56000),
    'wikizsl': ('1ELFGUIYDClmh9GrEHjFYoE_VI1t2a5nK', 113, 94383),
}
BERT_REV = '06fa25dc1b6e034e91607fd67c247051ccaeff65'
SBERT_REV = 'a85b0d10e16867eb768e411c66f79475505b65dd'
AUTHOR_REV = 'e2ffaea66244b2905055ee7cc47917985c3eab60'
PROPERTY_HASH = 'd9c91ac01f18398e58a48ad7913e3f168553e3c06b306879b1b2389bde3eb21d'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=SOURCES, required=True)
    p.add_argument('--m', type=int, choices=[5, 10, 15], required=True)
    p.add_argument('--run-id', type=int, choices=range(1, 6), required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--resume', default='')
    p.add_argument('--data-file', default='')
    p.add_argument('--data-source', choices=['snapshot', 'author-drive'], default='snapshot')
    p.add_argument('--expected-data-sha256', default='')
    p.add_argument('--allow-wiki-count-mismatch', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU before Save & Run All.')
    seed = 299 + args.run_id
    run_name = f'{args.dataset}_m{args.m}_seed{seed}' + ('_smoke' if args.smoke else '')
    work = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd() / 'outputs'
    task = work / 'temporary' / run_name
    out = work / run_name
    task.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    source_id, expected_relations, paper_samples = SOURCES[args.dataset]
    raw = Path(args.data_file) if args.data_file else task / f'{args.dataset}_raw.json'
    if args.data_file:
        source_metadata = dict(source='local override', author_identity_verified=False)
        if not raw.is_file():
            raise FileNotFoundError(f'Explicit DATA_FILE does not exist: {raw}')
    elif args.data_source == 'snapshot':
        source_metadata = prepare_snapshot(ROOT, args.dataset, raw)
        print('DATA SOURCE: versioned project snapshot; identity with author Drive file '
              'has not been independently verified.', flush=True)
    else:
        import gdown
        temporary = raw.with_suffix('.download.tmp')
        try:
            if not gdown.download(id=source_id, output=str(temporary), quiet=False):
                raise RuntimeError('Author dataset download failed.')
            # Never reuse a partial or HTML confirmation page as cached data.
            json.loads(temporary.read_text(encoding='utf-8'))
            temporary.replace(raw)
        except Exception as exc:
            raise RuntimeError('Author Drive download unavailable. Select DATA_SOURCE="snapshot" '
                               'to use the documented BTL project snapshot, or set DATA_FILE.') from exc
        finally:
            temporary.unlink(missing_ok=True)
        source_metadata = dict(source='author Google Drive link', source_id=source_id)
    raw_hash = digest(raw)
    if args.expected_data_sha256 and raw_hash != args.expected_data_sha256:
        raise ValueError('Dataset checksum differs from the locked experiment.')
    normalized = task / 'normalized.json'
    counts = preprocess(raw, normalized)
    print('DATA AUDIT:', json.dumps(dict(dataset=args.dataset, raw_sha256=raw_hash,
          samples=sum(counts.values()), relations=len(counts), paper_samples=paper_samples)), flush=True)
    if len(counts) != expected_relations:
        raise ValueError(f'Expected {expected_relations} relations, found {len(counts)}')
    if args.dataset == 'fewrel' and set(counts.values()) != {700}:
        raise ValueError('FewRel must have 700 samples per relation.')
    mismatch = sum(counts.values()) != paper_samples
    if mismatch and not args.allow_wiki_count_mismatch:
        raise ValueError(f'Dataset has {sum(counts.values())} samples; paper reports {paper_samples}. '
                         'Inspect source before explicitly allowing this discrepancy.')
    if args.smoke:
        data = json.loads(normalized.read_text(encoding='utf-8'))
        write(normalized, {k: v[:4] for k, v in data.items()})
    import urllib.request
    prop = task / 'author_properties.html'
    prop.write_bytes(urllib.request.urlopen(
        f'https://raw.githubusercontent.com/dinobby/ZS-BERT/{AUTHOR_REV}/resources/property_list.html',
        timeout=120).read())
    if digest(prop) != PROPERTY_HASH:
        raise ValueError('Author property descriptions checksum mismatch.')
    models = dict(allow_patterns=['*.json', 'vocab.txt', 'model.safetensors'],
                  ignore_patterns=['onnx/*', 'openvino/*'])
    bert = snapshot_download('google-bert/bert-large-cased', revision=BERT_REV, **models)
    os.environ['ZS_BERT_TOKENIZER_PATH'] = bert
    attrs = out / 'attributes.npz'
    descriptions = out / 'relation_descriptions.json'
    if args.resume:
        resume = Path(args.resume)
        for name in ['attributes.npz', 'relation_descriptions.json', 'experiment_config.json',
                     'active_checkpoint.pt', f'run_seed_{seed}.json', f'history_seed_{seed}.json',
                     f'train_seed_{seed}.log', 'provenance.json']:
            src = resume / name
            if src.exists() and src.resolve() != (out / name).resolve():
                if (out / name).exists():
                    raise FileExistsError(f'Resume destination already exists: {out / name}')
                shutil.copy2(src, out / name)
        if not attrs.exists() or not descriptions.exists() or not (out / 'experiment_config.json').exists():
            raise ValueError('Resume requires config, fixed attributes and descriptions.')
        if not (out / 'active_checkpoint.pt').exists() and not (out / f'run_seed_{seed}.json').exists():
            raise ValueError('Resume requires a checkpoint or completed result.')
    else:
        if (out / 'experiment_config.json').exists():
            raise FileExistsError('Existing run: use a fresh session or explicitly resume.')
        sbert = snapshot_download('sentence-transformers/bert-large-nli-mean-tokens',
                                  revision=SBERT_REV, **models)
        subprocess.run([sys.executable, str(ROOT/'scripts/build_attributes.py'),
                        str(normalized), str(prop), sbert, str(attrs), str(descriptions)], check=True)
    packages = {x: importlib.metadata.version(x) for x in
                ['torch', 'transformers', 'sentence-transformers', 'tokenizers', 'numpy', 'scikit-learn']}
    scientific = dict(dataset=args.dataset, m=args.m, epochs=1 if args.smoke else 10,
        batch_size=4, lr=5e-6, gamma=7.5, alpha=0.4, precision='fp32', dist_func='inner',
        eval_batch_size=4, gradient_checkpointing=True, bert_revision=BERT_REV,
        sbert_revision=SBERT_REV, author_revision=AUTHOR_REV, property_sha256=PROPERTY_HASH,
        data_sha256=digest(normalized), raw_data_sha256=raw_hash,
        attributes_sha256=digest(attrs), descriptions_sha256=digest(descriptions),
        source_hashes={str(f.relative_to(ROOT)): digest(f) for f in
                       [ROOT/'model/model.py', ROOT/'model/data_helper.py', ROOT/'model/evaluation.py',
                        ROOT/'scripts/train_run.py', ROOT/'scripts/normalize_data.py',
                        ROOT/'scripts/dataset_source.py', ROOT/'resources/datasets/manifest.json']},
        packages=packages, gpu_model=torch.cuda.get_device_name(0), smoke=args.smoke)
    config_hash = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    if args.resume:
        old = json.loads((out/'experiment_config.json').read_text(encoding='utf-8'))
        if old['config_hash'] != config_hash or old['seed'] != seed:
            raise ValueError('Resume config, dataset, seed or runtime differs.')
    cfg = dict(scientific, config_hash=config_hash, seed=seed, result_dir=str(out),
               attribute_path=str(attrs), data_path=str(normalized), bert_path=bert,
               checkpoint_every=1000, log_every=250, deadline_unix=args.deadline)
    config = out/'experiment_config.json'
    write(config, cfg)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    prior_sessions = []
    if (out/'provenance.json').exists():
        previous = json.loads((out/'provenance.json').read_text(encoding='utf-8'))
        prior_sessions = previous.get('sessions', [])
    write(out/'provenance.json', dict(commit=commit, source=source_metadata['source'],
        dataset_source_metadata=source_metadata, source_id=source_id,
        raw_data_sha256=raw_hash, samples=sum(counts.values()), relations=len(counts),
        paper_samples=paper_samples, count_discrepancy=mismatch,
        python=sys.version, cuda=torch.version.cuda, packages=packages, gpu=torch.cuda.get_device_name(0),
        seed=seed, dataset=args.dataset, m=args.m, smoke=args.smoke,
        sessions=prior_sessions + [dict(commit=commit, started_unix=time.time(), resumed=bool(args.resume))]))
    print('CONFIG:', json.dumps(cfg), flush=True)
    log_path = out/f'train_seed_{seed}.log'
    with open(log_path, 'a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/train_run.py'),
            '--config', str(config), '--seed', str(seed)], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in child.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = child.wait()
    if code:
        raise RuntimeError(f'Training failed ({code}); see {log_path}')
    manifest = {f.name: digest(f) for f in out.iterdir() if f.suffix in ['.json', '.log', '.npz']}
    write(out/'manifest.json', manifest)
    with zipfile.ZipFile(work/f'{run_name}_evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.suffix in ['.json', '.log', '.npz']:
                z.write(f, f'{run_name}/{f.name}')
    print('STATUS:', (out/'status.json').read_text(encoding='utf-8'), flush=True)
    print('Download evidence ZIP and executed notebook. Keep checkpoint on Kaggle if paused.')


if __name__ == '__main__':
    main()
