"""Verify published CASIA records and image integrity without redistributing data."""
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/casia_001'


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    manifest = json.loads((OUT / 'dataset_manifest.json').read_text())
    splits = {name: set(writers) for name, writers in manifest['splits'].items()}
    assert not (splits['train'] & splits['validation'] or splits['train'] & splits['test'] or splits['validation'] & splits['test'])
    writers = {row['writer_id']: row for row in manifest['writers']}
    for name, ids in splits.items():
        assert sum(writers[writer]['hanzi_samples'] for writer in ids) == manifest['sample_counts'][name]
    summary = json.loads((OUT / 'training_summary.json').read_text())
    train = read_jsonl(OUT / 'train_steps.jsonl')
    validation = read_jsonl(OUT / 'validation.jsonl')
    assert [row['step'] for row in train] == list(range(1, summary['steps'] + 1))
    best = min(validation, key=lambda row: row['total'])
    assert best['step'] == summary['best_validation_step']
    assert best['total'] == summary['best_validation_loss']
    assert summary['optimizer_examples'] == summary['steps'] * 2
    assert len(validation) == 11 and len(train) == 5000
    cases = {row['case_id']: row for row in json.loads((OUT / 'evaluation_cases.json').read_text())['cases']}
    assert len(cases) == 117
    checked = 0
    for phase in ['before', 'after']:
        rows = read_jsonl(OUT / f'eval_{phase}/samples.jsonl')
        assert len(rows) == len(cases) and {row['case_id'] for row in rows} == cases.keys()
        for row in rows:
            case = cases[row['case_id']]
            for key in ['writer', 'target', 'reference', 'seed', 'split', 'kind']:
                assert row[key] == case[key], (phase, row['case_id'], key)
            assert row['writer'] in splits[row['split']]
            assert row['target'] != row['reference']
            path = OUT / f'eval_{phase}' / row['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
            with Image.open(path) as im:
                assert im.size == (96, 96) and im.mode == 'RGB'
                im.verify()
            checked += 1
    sensitivity = json.loads((OUT / 'reference_sensitivity/metrics.json').read_text())
    for name, expected_hash in sensitivity['files_sha256'].items():
        path = OUT / 'reference_sensitivity' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        with Image.open(path) as im:
            im.verify()
    result = {'reference_sensitivity_generated_images': len(sensitivity['files_sha256']) - 1, 'status': 'passed', 'writer_disjoint': True, 'training_steps': len(train),
              'validation_records': len(validation), 'paired_cases': len(cases),
              'generated_images_verified': checked, 'checks': ['counts', 'checkpoint_selection', 'pairing', 'PNG_integrity', 'SHA256']}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
