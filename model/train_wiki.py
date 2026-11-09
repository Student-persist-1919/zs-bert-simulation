"""Training and Zero-Shot Evaluation script for ZS-BERT on Wiki-ZSL (NAACL 2021)."""

from argparse import ArgumentParser
import json
import os
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_helper import FewRelDataset, create_mini_batch, generate_attribute
from evaluation import evaluate, extract_relation_emb
from model import ZSBert


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def resolve_data_path(requested_path: str) -> str:
    candidates = [
        requested_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wiki_all.json"),
        os.path.join("data", "wiki_all.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"Could not find Wiki-ZSL dataset at '{requested_path}'. Run scripts/preprocess_wikizsl.py first."
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=300, help="random seed (paper default: 300)")
    parser.add_argument("--gamma", type=float, default=7.5, help="margin factor gamma (paper default: 7.5)")
    parser.add_argument("--alpha", type=float, default=0.4, help="balance coefficient alpha (paper default: 0.4)")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size (paper default: 4)")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs (paper default: 10)")
    parser.add_argument("--lr", type=float, default=5e-6, help="learning rate (paper default: 5e-6)")
    parser.add_argument("--n_unseen", type=int, default=10, help="number of unseen relations (paper default: 10)")
    parser.add_argument("--dist_func", type=str, default="inner", help="distance computing function: inner, euclidian, cosine")
    parser.add_argument("--data_path", type=str, default="../data/wiki_all.json", help="path to wiki_all.json")
    parser.add_argument("--eval_batch_size", type=int, default=256, help="evaluation batch size (default: 256)")
    parser.add_argument("--log_every", type=int, default=500, help="log detailed training metrics every N steps")
    parser.add_argument("--fp16", action="store_true", help="enable AMP mixed precision for faster training on T4 GPU")
    args, _ = parser.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    data_path = resolve_data_path(args.data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_train = json.load(f)

    all_keys = list(raw_train.keys())
    test_keys = random.sample(all_keys, k=args.n_unseen)
    train_keys = [k for k in all_keys if k not in test_keys]

    print(f"there are {len(train_keys)} kinds of relation in train.", flush=True)
    print(f"there are {len(test_keys)} kinds of relation in test.", flush=True)
    print(f"number of union of train and test: {len(set(train_keys) & set(test_keys))}", flush=True)

    property2idx, idx2property, pid2vec = generate_attribute(train_keys, test_keys)

    train_data = [item for k in train_keys for item in raw_train[k]]
    test_data = [item for k in test_keys for item in raw_train[k]]
    print(len(train_data), flush=True)
    print(len(test_data), flush=True)

    BERT_MODEL_NAME = "bert-large-cased"
    config = BertConfig.from_pretrained(BERT_MODEL_NAME, num_labels=len(set(train_keys)))
    config.pretrained_model_name = BERT_MODEL_NAME
    config.num_labels = len(set(train_keys))
    config.relation_emb_dim = 1024
    config.margin = args.gamma
    config.alpha = args.alpha
    config.dist_func = args.dist_func

    model = ZSBert.from_pretrained(BERT_MODEL_NAME, config=config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    model = model.to(device)

    trainset = FewRelDataset("train", train_data, pid2vec, property2idx)
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        collate_fn=create_mini_batch,
        shuffle=True,
    )

    test_y_attr = []
    for i in range(len(train_keys), len(train_keys) + len(test_keys)):
        test_y_attr.append(pid2vec[idx2property[i]])
    test_y_attr = np.array(test_y_attr)
    test_y = np.array([int(property2idx[i["relation"]]) for i in test_data])
    print(test_y_attr.shape, flush=True)
    print(test_y.shape, flush=True)

    testset = FewRelDataset("test", test_data, pid2vec, property2idx)
    testloader = DataLoader(
        testset,
        batch_size=args.eval_batch_size,
        collate_fn=create_mini_batch,
    )

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    best_p = 0.0
    best_r = 0.0
    best_f1 = 0.0
    total_steps = len(trainloader)

    for epoch in range(args.epochs):
        print(f"============== TRAIN ON THE {epoch + 1}-th EPOCH ==============", flush=True)
        running_loss = 0.0
        epoch_start = time.time()
        model.train()

        for step, data in enumerate(trainloader):
            tokens_tensors, segments_tensors, marked_e1, marked_e2, masks_tensors, relation_emb, labels = [
                t.to(device) for t in data
            ]
            optimizer.zero_grad()

            if args.fp16 and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs, out_relation_emb = model(
                        input_ids=tokens_tensors,
                        token_type_ids=segments_tensors,
                        e1_mask=marked_e1,
                        e2_mask=marked_e2,
                        attention_mask=masks_tensors,
                        input_relation_emb=relation_emb,
                        labels=labels,
                    )
                    loss = outputs[0]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, out_relation_emb = model(
                    input_ids=tokens_tensors,
                    token_type_ids=segments_tensors,
                    e1_mask=marked_e1,
                    e2_mask=marked_e2,
                    attention_mask=masks_tensors,
                    input_relation_emb=relation_emb,
                    labels=labels,
                )
                loss = outputs[0]
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            if step % 1000 == 0:
                print(f"[step {step}]" + "=" * (step // 1000), flush=True)

            if step > 0 and (step % args.log_every == 0 or step == total_steps - 1):
                elapsed = time.time() - epoch_start
                done = step + 1
                speed = done / max(elapsed, 1e-6)
                eta = (total_steps - done) / max(speed, 1e-6)
                pct = (done / total_steps) * 100.0
                print(
                    f"  [Epoch {epoch + 1:02d}/{args.epochs:02d} | Step {done:5d}/{total_steps} ({pct:5.1f}%)] "
                    f"loss: {running_loss / done:.4f} | speed: {speed:.1f} step/s | eta: {format_time(eta)}",
                    flush=True,
                )

        print("============== EVALUATION ON TEST DATA ==============", flush=True)
        preds = extract_relation_emb(model, testloader).cpu().numpy()
        pt, rt, f1t = evaluate(
            preds,
            test_y_attr,
            test_y,
            idx2property,
            len(train_keys),
            args.dist_func,
        )
        print(f"[testing performance] precision: {pt:.4f}, recall: {rt:.4f}, f1 score: {f1t:.4f}", flush=True)

        if f1t > best_f1:
            best_p = pt
            best_r = rt
            best_f1 = f1t

        print(f"[best] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
