from typing import List, Dict, Any, Optional
from .pipeline import ThreatPipeline
from .evaluation import evaluate_events, build_labeled_windows
from .audit import audit_prediction, create_snapshot
from .drift import simulate_slang_shift, simulate_missing_data
from .adversarial import adversarial_variations
from .uncertainty import entropy_from_probs, ensemble_disagreement, label_uncertainty


def evaluate_historical(documents: List[Any], ground_truth: Dict[str, Dict[str, Any]], db_path: Optional[str] = None):
    pipeline = ThreatPipeline()
    # Run pipeline
    results = pipeline.run(documents)
    events = [type('E', (), e) if isinstance(e, dict) else e for e in results.get('events', [])]
    # normalize event_alerts back into objects if dicts
    event_alerts = {ea['id']: type('A', (), ea) if isinstance(ea, dict) else ea for ea in results.get('event_alerts', [])}

    metrics, records = evaluate_events(events, event_alerts, ground_truth)

    # audit snapshot
    if db_path:
        audit_prediction(db_path, event_id="*", action="historical_evaluate", payload={"metrics": metrics})
    # also write a disk snapshot
    create_snapshot("evaluation_snapshot.json", {"metrics": metrics, "records_sample": records[:20]})
    return metrics, records


def simulate_and_evaluate(documents: List[Dict[str, Any]], ground_truth: Dict[str, Dict[str, Any]], slang_map: Dict[str, str], code_map: Dict[str, str]):
    # Apply drift and adversarial perturbations and re-evaluate
    drifted = simulate_slang_shift(documents, slang_map)
    drifted = simulate_missing_data(drifted, drop_rate=0.05)
    adv = adversarial_variations(drifted, code_map)
    return evaluate_historical(adv, ground_truth)
