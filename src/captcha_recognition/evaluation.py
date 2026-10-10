"""Case-sensitive sequence evaluation; validation is distinct from diagnostic data."""
import json
from pathlib import Path
import torch


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def edit_distance(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, row[j]+1, row[j-1]+int(x != y)))
        row = current
    return row[-1]


def summarize(rows, predictions, details=False):
    if len(rows) != len(predictions):
        raise ValueError('Prediction count mismatch')
    groups = {}
    for row, text in zip(rows, predictions):
        group = groups.setdefault(row['group'], dict(samples=0, correct=0, characters=0, edits=0, predictions=[]))
        group['samples'] += 1
        group['correct'] += int(row['label'] == text)
        group['characters'] += len(row['label'])
        group['edits'] += edit_distance(row['label'], text)
        if details:
            group['predictions'].append(dict(file=row['path'], label=row['label'], prediction=text))
    for group in groups.values():
        group['exact_acc'] = group['correct']/group['samples']
        group['cer'] = group['edits']/group['characters']
    public = [group['exact_acc'] for name, group in groups.items() if name.startswith('public_')]
    synthetic = [group['exact_acc'] for name, group in groups.items() if not name.startswith('public_')]
    return dict(evaluation=groups, macro_exact=sum(g['exact_acc'] for g in groups.values())/len(groups),
        balanced_validation_score=(.5*sum(public)/len(public)+.5*sum(synthetic)/len(synthetic)) if public and synthetic else None)


@torch.inference_mode()
def evaluate(model, data, batch_size=32, details=False, mode='all'):
    model.eval()
    variants = ['ctc', 'attention', 'hybrid'] if mode == 'all' else [mode]
    predictions = {name: [] for name in variants}
    for offset in range(0, len(data.rows), batch_size):
        results = model.recognize(data.images[offset:offset+batch_size], data.widths[offset:offset+batch_size], mode)
        for name in variants:
            predictions[name].extend(row['prediction'] if name == 'hybrid' or mode != 'all' else row[name] for row in results)
    return {name: summarize(data.rows, texts, details) for name, texts in predictions.items()}
