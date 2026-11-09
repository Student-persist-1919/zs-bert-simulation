"""Data loading, attribute generation, and Dataset classes for ZS-BERT (decompiled from dinobby/ZS-BERT)."""

from html.parser import HTMLParser
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class _PropertyTableParser(HTMLParser):
    """Fallback HTML table parser for property_list.html."""

    def __init__(self):
        super().__init__()
        self.in_tr = False
        self.in_td = False
        self.current_row = []
        self.current_cell = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.in_tr = True
            self.current_row = []
        elif tag == "td" and self.in_tr:
            self.in_td = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag == "td" and self.in_td:
            self.in_td = False
            self.current_row.append("".join(self.current_cell).strip())
        elif tag == "tr" and self.in_tr:
            self.in_tr = False
            if len(self.current_row) >= 3:
                self.rows.append(self.current_row)

    def handle_data(self, data):
        if self.in_td:
            self.current_cell.append(data)


def load_datas(json_files, val_portion=0.0, load_vertices=True):
    """
    Load semantic graphs from multiple json files and if specified reserve a portion of the data for validation.

    :param json_files: list of input json files
    :param val_portion: a portion of the data to reserve for validation
    :return: a tuple of the data and validation data
    """
    data = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            if load_vertices:
                data = data + json.load(f)
            else:
                data = data + json.load(f)
    print("Loaded data size:", len(data), flush=True)

    val_data = []
    if val_portion > 0.0:
        val_size = int(len(data) * val_portion)
        rest_size = len(data) - val_size
        val_data = data[rest_size:]
        data = data[:rest_size]
        print("Training and dev set sizes:", (len(data), len(val_data)), flush=True)
    return data, val_data


def load_data(json_file, val_portion=0.0, load_vertices=True):
    return load_datas([json_file], val_portion, load_vertices)


def split_wiki_data(data, mask_relation):
    train_data = []
    test_data = []
    for i in data:
        kbID = i["edgeSet"][0]["kbID"]
        if kbID not in mask_relation:
            train_data.append(i)
        else:
            test_data.append(i)
    return train_data, test_data


def generate_attribute(
    train_label,
    test_label,
    att_dim=1024,
    prop_list_path="../resources/property_list.html",
):
    from sentence_transformers import SentenceTransformer

    property2idx = {}
    idx2property = {}
    idx = 0
    for i in set(train_label):
        property2idx[i] = idx
        idx2property[idx] = i
        idx += 1
    for i in set(test_label):
        property2idx[i] = idx
        idx2property[idx] = i
        idx += 1

    if not os.path.exists(prop_list_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prop_list_path = os.path.join(base_dir, "resources", "property_list.html")

    try:
        prop_list = pd.read_html(prop_list_path)[0]
    except Exception:
        with open(prop_list_path, "r", encoding="utf-8") as f:
            parser = _PropertyTableParser()
            parser.feed(f.read())
        prop_list = pd.DataFrame(
            [r[:3] for r in parser.rows], columns=["ID", "label", "description"]
        )

    prop_list = prop_list.loc[prop_list.ID.isin(property2idx.keys())]
    encoder = SentenceTransformer("bert-large-nli-mean-tokens")
    sentence_embeddings = encoder.encode(prop_list.description.to_list())

    if att_dim < 1024:
        from sklearn.decomposition import TruncatedSVD

        print(f"att_dim={att_dim}", flush=True)
        svd = TruncatedSVD(n_components=att_dim, n_iter=10, random_state=42)
        sentence_embeddings = svd.fit_transform(sentence_embeddings)
        print(f"size of sentence_embeddings: {sentence_embeddings.shape}", flush=True)

    pid2vec = {}
    for pid, embedding in zip(prop_list.ID, sentence_embeddings):
        pid2vec[pid] = embedding.astype("float32")

    return property2idx, idx2property, pid2vec


def mark_wiki_entity(edge, sent_len):
    e1 = np.array(edge["left"]) + 1
    e2 = np.array(edge["right"]) + 1
    e1 = e1[e1 < sent_len]
    e2 = e2[e2 < sent_len]
    marked_e1 = np.array([0] * sent_len)
    marked_e2 = np.array([0] * sent_len)
    if len(e1) > 0:
        marked_e1[e1] += 1
    if len(e2) > 0:
        marked_e2[e2] += 1
    return (
        torch.tensor(marked_e1, dtype=torch.long),
        torch.tensor(marked_e2, dtype=torch.long),
    )


def mark_fewrel_entity(edge, sent_len):
    e1 = np.array(edge["h"][2][0]) + 1
    e2 = np.array(edge["t"][2][0]) + 1
    e1 = e1[e1 < sent_len]
    e2 = e2[e2 < sent_len]
    marked_e1 = np.array([0] * sent_len)
    marked_e2 = np.array([0] * sent_len)
    if len(e1) > 0:
        marked_e1[e1] += 1
    if len(e2) > 0:
        marked_e2[e2] += 1
    return (
        torch.tensor(marked_e1, dtype=torch.long),
        torch.tensor(marked_e2, dtype=torch.long),
    )


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    marked_e1 = [s[2] for s in samples]
    marked_e2 = [s[3] for s in samples]
    relation_emb = [s[4] for s in samples]
    if samples[0][5] is not None:
        label_ids = torch.stack([s[5] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    marked_e1 = pad_sequence(marked_e1, batch_first=True)
    marked_e2 = pad_sequence(marked_e2, batch_first=True)
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    relation_emb = torch.from_numpy(np.asarray(relation_emb, dtype=np.float32))

    return (
        tokens_tensors,
        segments_tensors,
        marked_e1,
        marked_e2,
        masks_tensors,
        relation_emb,
        label_ids,
    )


class WikiDataset(Dataset):
    def __init__(self, mode, data, pid2vec, property2idx):
        assert mode in ["train", "test"]
        self.mode = mode
        self.data = data
        self.pid2vec = pid2vec
        self.property2idx = property2idx
        self.len = len(data)
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

    def __getitem__(self, idx):
        g = self.data[idx]
        sentence = " ".join(g["tokens"])
        tokens = self.tokenizer.tokenize(sentence)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
        tokens_tensor = torch.tensor(tokens_ids)
        segments_tensor = torch.tensor([0] * len(tokens_ids), dtype=torch.long)
        edge = g["edgeSet"][0]
        marked_e1, marked_e2 = mark_wiki_entity(edge, len(tokens_ids))
        property_kbid = edge["kbID"]
        relation_emb = self.pid2vec[property_kbid]
        if self.mode == "train":
            label = int(self.property2idx[property_kbid])
            label_tensor = torch.tensor(label)
        elif self.mode == "test":
            label_tensor = None
        return (
            tokens_tensor,
            segments_tensor,
            marked_e1,
            marked_e2,
            relation_emb,
            label_tensor,
        )

    def __len__(self):
        return self.len


class FewRelDataset(Dataset):
    def __init__(self, mode, data, pid2vec, property2idx):
        assert mode in ["train", "test"]
        self.mode = mode
        self.data = data
        self.pid2vec = pid2vec
        self.property2idx = property2idx
        self.len = len(data)
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        # Pre-cache tokenized items once to avoid re-tokenizing on CPU every epoch
        self._cache = [self._encode_item(i) for i in range(self.len)]

    def _encode_item(self, idx):
        g = self.data[idx]
        sentence = " ".join(g["tokens"])
        tokens = self.tokenizer.tokenize(sentence)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
        tokens_tensor = torch.tensor(tokens_ids)
        segments_tensor = torch.tensor([0] * len(tokens_ids), dtype=torch.long)
        marked_e1, marked_e2 = mark_fewrel_entity(g, len(tokens_ids))
        relation_emb = self.pid2vec[g["relation"]]
        if self.mode == "train":
            label = int(self.property2idx[g["relation"]])
            label_tensor = torch.tensor(label)
        elif self.mode == "test":
            label_tensor = None
        return (
            tokens_tensor,
            segments_tensor,
            marked_e1,
            marked_e2,
            relation_emb,
            label_tensor,
        )

    def __getitem__(self, idx):
        return self._cache[idx]

    def __len__(self):
        return self.len
