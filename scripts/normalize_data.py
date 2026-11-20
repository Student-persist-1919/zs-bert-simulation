"""Normalize the user's FewRel format without guessing native index lists as spans."""
from collections import defaultdict
import json
from pathlib import Path


def normalize_entity(entity, tokens, dict_position_format='span'):
    if isinstance(entity, dict):
        name, qid, positions = entity.get('name', ''), entity.get('id', 'Q0'), entity['pos']
        if positions and isinstance(positions[0], list):
            indices = positions[0]  # explicitly nested index list
        elif dict_position_format == 'span':
            if len(positions) != 2:
                raise ValueError('Expected dict.pos=[start,end_exclusive]; set DICT_POSITION_FORMAT="indices" if needed.')
            indices = list(range(*positions))
        else:
            indices = positions
    else:
        name, qid, positions = entity
        indices = positions[0]  # official FewRel [name,id,[[token indices], ...]]
    if not indices or any(type(i) is not int or not 0 <= i < len(tokens) for i in indices):
        raise ValueError(f'Invalid entity positions: {indices!r}; token count={len(tokens)}')
    if indices != sorted(set(indices)):
        raise ValueError(f'Entity positions must be unique and sorted: {indices!r}')
    return [name, qid, [indices]]


def preprocess(source, destination, dict_position_format='span'):
    raw = json.loads(Path(source).read_text(encoding='utf-8'))
    if isinstance(raw, dict):
        rows = ((pid, item) for pid, items in raw.items() for item in items)
    elif isinstance(raw, list):
        rows = ((item['relation'], item) for item in raw)
    else:
        raise ValueError('Expected relation->instances dictionary or flat instance list.')
    grouped = defaultdict(list)
    for pid, item in rows:
        tokens = item.get('tokens', item.get('token'))
        if not tokens or not all(isinstance(token, str) for token in tokens):
            raise ValueError('Each sample must contain a nonempty list of tokens.')
        grouped[pid].append({'tokens': tokens, 'relation': pid,
                             'h': normalize_entity(item['h'], tokens, dict_position_format),
                             't': normalize_entity(item['t'], tokens, dict_position_format)})
    counts = {pid: len(items) for pid, items in grouped.items()}
    # Dataset-specific counts are checked by kaggle_run after normalization.
    Path(destination).write_text(json.dumps(grouped, ensure_ascii=False), encoding='utf-8')
    return counts


