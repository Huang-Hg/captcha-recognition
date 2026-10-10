"""Freeze a candidate list, select on validation, then run the chosen diagnostic."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from .evaluation import dump
from .provenance import capture_source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    capture_source(out/'configs/source_snapshot')
    validation_sha = hashlib.sha256(Path(spec['validation']).read_bytes()).hexdigest()
    names = [row['name'] for row in spec['candidates']]
    if len(set(names)) != len(names) or any(not name.replace('_', '').isalnum() for name in names):
        raise ValueError('Candidate names must be unique path-safe identifiers')
    for row in spec['candidates']:
        row['checkpoint_sha256'] = hashlib.sha256(Path(row['checkpoint']).read_bytes()).hexdigest()
    dump(out/'configs/preregistered.json', dict(spec=spec, validation_sha256=validation_sha,
        selection='balanced real/synthetic macro exact, validation only', precision='fp32', batch_size=8,
        test_selection=False, tie_rule='First supplied candidate'))

    def run(row, manifest, destination):
        subprocess.run([sys.executable, '-m', 'captcha_recognition.evaluate', '--checkpoint', row['checkpoint'],
            '--mode', row['mode'], '--manifest', manifest, '--output', str(destination), '--batch-size', '8'], check=True)
        return json.loads(destination.read_text())

    def reuse(row, manifest, source, destination):
        result = json.loads(Path(source).read_text())
        decoder = {'rgb_channel_ctc_mixture': 'ctc_multiview', 'greedy': 'ctc'}.get(result['decoder'], result['decoder'])
        manifest_sha = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        if (result['checkpoint_sha256'] != row['checkpoint_sha256'] or result['manifest_sha256'] != manifest_sha
                or decoder != row['mode'] or result['precision'] != 'fp32' or result['batch_size'] != 8
                or result.get('research_input_size') is not None):
            raise ValueError('Cached evaluation contract mismatch')
        dump(destination, dict(result, reused_from=str(source)))
        return result

    for row in spec['candidates']:
        destination = out/f'logs/validation_{row["name"]}.json'
        if row.get('cached_validation'):
            result = reuse(row, spec['validation'], row['cached_validation'], destination)
        else:
            result = run(row, spec['validation'], destination)
        public = [g['exact_acc'] for name, g in result['evaluation'].items() if name.startswith('public_')]
        synthetic = [g['exact_acc'] for name, g in result['evaluation'].items() if not name.startswith('public_')]
        if not public or not synthetic:
            raise ValueError('Incomplete validation domains')
        row['validation_score'] = .5*sum(public)/len(public)+.5*sum(synthetic)/len(synthetic)
        print('CANDIDATE', row['name'], row['validation_score'], flush=True)
    winner = max(spec['candidates'], key=lambda row: row['validation_score'])
    dump(out/'configs/selection_lock.json', dict(selected=winner, candidates=spec['candidates'], test_selection=False))
    destination = out/'logs/observed_diagnostic.json'
    if winner.get('cached_diagnostic'):
        reuse(winner, spec['diagnostic'], winner['cached_diagnostic'], destination)
    else:
        run(winner, spec['diagnostic'], destination)
    dump(out/'logs/status.json', dict(state='complete', selected=winner['name'], production_ready=False))
    print('SELECTED', winner, flush=True)


if __name__ == '__main__':
    main()
