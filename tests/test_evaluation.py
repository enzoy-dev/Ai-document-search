import json
from pathlib import Path

import pytest

from app.document_processor import process_document
from experiments.evaluate_retrieval import score_cases


def test_scoring_metrics_handle_ranking_rejection_and_partial_recall():
    cases = [{'question': 'one', 'expected': [{'filename': 'a.pdf', 'page': 1}, {'filename': 'a.pdf', 'page': 2}]},
             {'question': 'two', 'expected': []}]
    ranked = [[{'source': 'b.pdf', 'page': 1, 'distance': 0.1}, {'source': 'a.pdf', 'page': 1, 'distance': 0.8}],
              [{'source': 'b.pdf', 'page': 1, 'distance': 2.0}]]
    result = score_cases(cases, ranked, threshold=0.5, top_k=2)
    assert result['recall_at_k'] == 0.5
    assert result['accepted_recall_at_k'] == 0
    assert result['mrr_at_k'] == 0.5
    assert result['unanswerable_rejection_rate'] == 1
    assert score_cases(cases, ranked, threshold=3, top_k=2)['unanswerable_rejection_rate'] == 0


def test_benchmark_evidence_exists_in_real_pdfs():
    directory = Path(__file__).resolve().parents[1] / 'evaluations'
    data = json.loads((directory / 'retrieval.json').read_text())
    pages = {(Path(name).name, chunk['page']) for name in data['documents']
             for chunk in process_document(str(directory / name))['chunks']}
    assert len(data['cases']) == 20 and len(pages) == 8
    assert any(not case['expected'] for case in data['cases'])
    assert all((expected['filename'], expected['page']) in pages for case in data['cases'] for expected in case['expected'])


def test_misaligned_scoring_is_rejected():
    with pytest.raises(ValueError):
        score_cases([{'question': 'q', 'expected': []}], [], 1.2, 4)


def test_committed_reports_match_the_dataset_and_record_actual_weight_commits():
    import hashlib
    directory = Path(__file__).resolve().parents[1] / 'evaluations'
    path = directory / 'retrieval.json'
    data = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes())
    for name in data['documents']:
        digest.update((directory / name).read_bytes())
    reports = list((directory / 'baselines').glob('*.json'))
    assert len(reports) == 2
    for path in reports:
        report = json.loads(path.read_text())
        assert report['dataset_sha256'] == digest.hexdigest()
        assert len(report['resolved_model_identity']) == 40
        assert report['metric'] == 'normalized_squared_l2'
        assert len(report['results']) == report['cases'] == 20
        cases = [{'question': row['question'], 'expected': row['expected']} for row in report['results']]
        metrics = score_cases(cases, [row['ranked'] for row in report['results']], report['threshold'], report['top_k'])
        for name in ['recall_at_k', 'accepted_recall_at_k', 'mrr_at_k', 'unanswerable_rejection_rate']:
            assert report[name] == pytest.approx(metrics[name])
