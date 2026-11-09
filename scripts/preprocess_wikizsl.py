"""Preprocess Wiki-ZSL dataset (wikizsl_dataset.json -> data/wiki_all.json) for ZS-BERT."""

import argparse
from collections import defaultdict
import json
import os


def convert_span_to_indices(pos, sent_len):
    if not pos:
        return [0]
    if isinstance(pos[0], list):
        pos = pos[0]
    if len(pos) == 2 and pos[0] < pos[1] <= sent_len:
        indices = list(range(pos[0], pos[1]))
    else:
        indices = [i for i in pos if 0 <= i < sent_len]
    return indices if indices else [0]


def preprocess_wikizsl(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[Preprocess] Reading Wiki-ZSL raw dataset from: {input_path}", flush=True)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        flat_list = []
        for rel, items in raw_data.items():
            for item in items:
                item["relation"] = rel
                flat_list.append(item)
        raw_data = flat_list

    wiki_dict = defaultdict(list)
    total_samples = 0

    for item in raw_data:
        rel = item["relation"]
        tokens = item.get("token", item.get("tokens", []))
        sent_len = len(tokens)

        if isinstance(item["h"], dict):
            h_name = item["h"].get("name", "")
            if isinstance(h_name, list):
                h_name = " ".join(h_name)
            h_id = item["h"].get("id", "Q0")
            h_indices = convert_span_to_indices(item["h"]["pos"], sent_len)
            h_list = [h_name, h_id, [h_indices]]
        else:
            h_indices = convert_span_to_indices(item["h"][2][0], sent_len)
            h_list = [item["h"][0], item["h"][1], [h_indices]]

        if isinstance(item["t"], dict):
            t_name = item["t"].get("name", "")
            if isinstance(t_name, list):
                t_name = " ".join(t_name)
            t_id = item["t"].get("id", "Q0")
            t_indices = convert_span_to_indices(item["t"]["pos"], sent_len)
            t_list = [t_name, t_id, [t_indices]]
        else:
            t_indices = convert_span_to_indices(item["t"][2][0], sent_len)
            t_list = [item["t"][0], item["t"][1], [t_indices]]

        new_item = {
            "tokens": tokens,
            "h": h_list,
            "t": t_list,
            "relation": rel,
        }
        wiki_dict[rel].append(new_item)
        total_samples += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(wiki_dict, f)

    print(
        f"[Preprocess] Successfully converted {total_samples} samples across "
        f"{len(wiki_dict)} relations -> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Wiki-ZSL dataset for ZS-BERT")
    parser.add_argument(
        "--input",
        type=str,
        default="wikizsl_dataset.json",
        help="Path to raw wikizsl_dataset.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wiki_all.json",
        help="Path to output data/wiki_all.json",
    )
    args = parser.parse_args()
    preprocess_wikizsl(args.input, args.output)
