"""Run one independently initialized FewRel experiment; invoked by each notebook."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from data_helper import FewRelDataset, create_mini_batch
from evaluation import compute_macro_PRF
from model import ZSBert


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def metrics(model, loader, attributes, gold, offset, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        attr = torch.as_tensor(attributes, device=device)
        for data in loader:
            ids, segments, e1, e2, mask, rel, _ = data
            _, emb = model(input_ids=ids.to(device), token_type_ids=segments.to(device),
                           e1_mask=e1.to(device), e2_mask=e2.to(device),
                           attention_mask=mask.to(device))
            # Exact exhaustive nearest neighbour under negative inner product.
            # Negative inner product is not a metric; BallTree pruning is invalid.
            predictions.extend((emb @ attr.T).argmax(1).cpu().numpy() + offset)
    predictions = np.asarray(predictions)
    p, r, f1 = compute_macro_PRF(predictions, gold)
    from sklearn.metrics import precision_recall_fscore_support
    sp, sr, sf, _ = precision_recall_fscore_support(
        gold, predictions, labels=np.arange(offset, offset + len(attributes)),
        average='macro', zero_division=0)
    return {'precision': float(p), 'recall': float(r), 'f1': float(f1),
            'standard_macro_precision': float(sp), 'standard_macro_recall': float(sr),
            'standard_macro_f1': float(sf)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    seed = args.seed
    out = Path(cfg['result_dir'])
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f'run_seed_{seed}.json'
    checkpoint_file = out / 'active_checkpoint.pt'
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding='utf-8'))
        if result.get('complete') and result['config_hash'] == cfg['config_hash']:
            write_json(out / 'status.json', {'status': 'complete', 'seed': seed,
                       'config_hash': cfg['config_hash']})
            print(f'Seed {seed}: completed earlier, skipping.', flush=True)
            return
        raise RuntimeError(f'Existing result has a different config: {result_file}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('Select a Kaggle GPU accelerator before running training.')
    device = torch.device('cuda:0')
    checkpoint = None
    if checkpoint_file.exists():
        # Only load locally produced/trusted notebook outputs, never arbitrary inputs.
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        if checkpoint['config_hash'] != cfg['config_hash'] or checkpoint['seed'] != seed:
            raise RuntimeError('Active checkpoint does not match config/seed. Restore all run JSONs too.')

    raw = json.loads(Path(cfg['data_path']).read_text(encoding='utf-8'))
    all_keys = list(raw)  # Preserve the input's relation order, as in the original script.
    test_keys = random.sample(all_keys, cfg['m'])
    train_keys = [key for key in all_keys if key not in test_keys]
    assert not set(train_keys).intersection(test_keys)
    # Deterministic label order; no Python set iteration dependence.
    labels = train_keys + test_keys
    property2idx = {pid: i for i, pid in enumerate(labels)}
    attrs = np.load(cfg['attribute_path'])
    pid2vec = {pid: attrs[pid] for pid in labels}
    train_data = [item for key in train_keys for item in raw[key]]
    test_data = [item for key in test_keys for item in raw[key]]
    print(f'm={cfg["m"]}, seed={seed}, train={len(train_data)}, test={len(test_data)}', flush=True)
    print('Unseen relations:', test_keys, flush=True)
    trainset = FewRelDataset('train', train_data, pid2vec, property2idx)
    testset = FewRelDataset('test', test_data, pid2vec, property2idx)
    testloader = DataLoader(testset, batch_size=cfg['eval_batch_size'],
                            collate_fn=create_mini_batch, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
    test_attr = np.stack([pid2vec[key] for key in test_keys])
    gold = np.array([property2idx[item['relation']] for item in test_data])
    config = BertConfig.from_pretrained(cfg['bert_path'], num_labels=len(train_keys))
    config.relation_emb_dim = 1024
    config.margin, config.alpha, config.dist_func = cfg['gamma'], cfg['alpha'], 'inner'
    # Original implementation uses explicit attention, before SDPA became the default.
    model = ZSBert.from_pretrained(cfg['bert_path'], config=config,
                                  attn_implementation='eager').to(device)
    if cfg['gradient_checkpointing']:
        model.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    start_epoch, next_step, totals, history, prior_seconds = 0, 0, [0., 0], [], 0.
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, next_step = checkpoint['epoch'], checkpoint['next_step']
        totals, history = checkpoint['totals'], checkpoint['history']
        prior_seconds = checkpoint['elapsed_sec']
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        print(f'Resuming epoch {start_epoch + 1}, next batch {next_step + 1}', flush=True)
        del checkpoint
    started = time.time()

    def save_checkpoint(epoch, step, running):
        tmp = checkpoint_file.with_suffix('.tmp')
        state = {'config_hash': cfg['config_hash'], 'seed': seed,
                 'epoch': epoch, 'next_step': step, 'totals': running, 'history': history,
                 'elapsed_sec': prior_seconds + time.time() - started,
                 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        torch.save(state, tmp)
        tmp.replace(checkpoint_file)
        print(f'Checkpoint saved (epoch={epoch + 1}, next_step={step}).', flush=True)

    def pause_if_due(epoch, step, running):
        if time.time() < cfg['deadline_unix']:
            return False
        save_checkpoint(epoch, step, running)
        write_json(out / 'status.json', {'status': 'paused', 'seed': seed,
                   'config_hash': cfg['config_hash'], 'epoch': epoch, 'next_step': step})
        print('PAUSED: time budget reached; resume this seed in another session.', flush=True)
        return True

    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        if epoch != start_epoch:
            next_step, totals = 0, [0., 0]
        # Fixed permutation per epoch allows skipping batches on resume without replaying dropout.
        gen = torch.Generator().manual_seed(seed * 10000 + epoch)
        order = torch.randperm(len(trainset), generator=gen).tolist()
        loader = DataLoader(trainset, batch_size=cfg['batch_size'],
                            sampler=order[next_step * cfg['batch_size']:],
                            collate_fn=create_mini_batch, num_workers=0,
                            generator=torch.Generator().manual_seed(seed * 10000 + epoch))
        total_steps = (len(trainset) + cfg['batch_size'] - 1) // cfg['batch_size']
        for step, data in enumerate(loader, start=next_step):
            if pause_if_due(epoch, step, totals):
                return
            ids, segments, e1, e2, mask, rel, y = [t.to(device) for t in data]
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = model(input_ids=ids, token_type_ids=segments, e1_mask=e1,
                               e2_mask=e2, attention_mask=mask,
                               input_relation_emb=rel, labels=y)
            loss = outputs[0]
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch+1}, batch {step+1}')
            loss.backward()
            optimizer.step()
            totals[0] += loss.item()
            totals[1] += 1
            if (step + 1) % cfg['log_every'] == 0 or step + 1 == total_steps:
                print(f'seed={seed} epoch={epoch+1}/{cfg["epochs"]} '
                      f'batch={step+1}/{total_steps} loss={totals[0]/totals[1]:.6f}', flush=True)
            if (step + 1) % cfg['checkpoint_every'] == 0 and step + 1 < total_steps:
                save_checkpoint(epoch, step + 1, totals)
        if pause_if_due(epoch, total_steps, totals):
            return
        scores = metrics(model, testloader, test_attr, gold, len(train_keys), device)
        row = {'epoch': epoch + 1, 'train_loss': totals[0] / max(totals[1], 1), **scores}
        history.append(row)
        print('EVALUATION:', json.dumps(row), flush=True)
        write_json(out / f'history_seed_{seed}.json', history)
        save_checkpoint(epoch + 1, 0, [0., 0])
    best = max(history, key=lambda row: row['f1'])
    result = {'complete': True, 'config_hash': cfg['config_hash'], 'seed': seed, 'm': cfg['m'],
              'train_relations': train_keys, 'test_relations': test_keys,
              'train_samples': len(train_data), 'test_samples': len(test_data),
              'best': best, 'final': history[-1], 'history': history,
              'total_time_sec': prior_seconds + time.time() - started,
              'metric_note': 'Main P/R/F1 use authors compute_macro_PRF; standard macro metrics also saved.',
              'selection_note': 'best uses unseen TEST F1, matching authors script; final does not select on test.',
              'runtime': {p: importlib.metadata.version(p) for p in
                          ['torch', 'transformers', 'sentence-transformers', 'tokenizers', 'numpy', 'scikit-learn']},
              'gpu': torch.cuda.get_device_name(0)}
    write_json(result_file, result)
    write_json(out / 'status.json', {'status': 'complete', 'seed': seed,
               'config_hash': cfg['config_hash']})
    checkpoint_file.unlink(missing_ok=True)
    print(f'COMPLETE seed={seed}: best={best["f1"]:.6f}, final={history[-1]["f1"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
