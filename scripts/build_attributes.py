import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
labels = list(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
table = pd.read_html(sys.argv[2])[0]
descriptions = {}
for pid in labels:
    rows = table.loc[table.ID == pid, "description"]
    if len(rows) != 1 or not isinstance(rows.iloc[0], str) or not rows.iloc[0].strip():
        raise ValueError(f"Missing/ambiguous description: {pid}")
    descriptions[pid] = rows.iloc[0]
encoder = SentenceTransformer(sys.argv[3], device="cuda", model_kwargs={"attn_implementation": "eager"})
encoder.eval()
emb = encoder.encode([descriptions[k] for k in labels], batch_size=4,
                     convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
assert emb.shape == (len(labels), 1024) and np.isfinite(emb).all()
np.savez(sys.argv[4], **{pid: row.astype("float32") for pid, row in zip(labels, emb)})
Path(sys.argv[5]).write_text(json.dumps(descriptions, ensure_ascii=False, indent=2), encoding="utf-8")
