from datetime import datetime
from typing import Dict, List, Optional

from .entities import extract_entities, detect_weapons
from .events import group_documents_into_events
from .event_cluster import cluster_documents
from .graph import KnowledgeGraph
from .ingest import unify_documents
from .models import Document, Entity, KnowledgeFact
from .risk import score_risk, score_event
from .timeline import reconstruct_timeline
from .prediction import predict_for_top_people, predict_group_trajectory, simulate_graph_change, suggest_interventions, compute_sentiment
from .uncertainty import entropy_from_probs, ensemble_disagreement, label_uncertainty
from .digital_twin import generate_beliefs_from_graph, generate_competing_hypotheses
from .models_ml import HeuristicPredictor
from .audit import audit_prediction, create_snapshot
from .evaluation import evaluate_events
from .decision import decide_event
from .task_queue import enqueue_for_event


class ThreatPipeline:
    def __init__(self):
        self.graph = KnowledgeGraph()

    def run(self, documents: List[Document], persist_db_path: str = None, evaluate: bool = False, ground_truth: dict = None, audit_db_path: str = None) -> dict:
        normalized_documents = unify_documents(documents)
        document_entities: Dict[str, List[Entity]] = {}
        all_entities: List[Entity] = []
        alerts = []

        for doc in normalized_documents:
            entities = extract_entities(doc)
            entities.extend(detect_weapons(doc.text))
            document_entities[doc.id] = entities
            all_entities.extend(entities)

            for entity in entities:
                self.graph.add_entity(entity)

            alert = score_risk(doc, entities)
            alerts.append(alert)

            for entity in entities:
                for other in entities:
                    if entity is not other:
                        self.graph.connect_entities(entity, other, relation="related_to")

        # form clusters using embeddings + time window
        clusters = cluster_documents(normalized_documents)

        # convert clusters to Event objects using existing grouping logic
        events = []
        for cluster_docs in clusters:
            # use grouping to get a single Event per cluster
            sub_events = group_documents_into_events(cluster_docs, document_entities)
            # group_documents_into_events may return multiple events for a cluster; merge
            events.extend(sub_events)

        for event in events:
            self.graph.connect_event(event)

        # compute event-level alerts
        documents_by_id = {d.id: d for d in normalized_documents}
        event_alerts = [score_event(e, documents_by_id, document_entities) for e in events]
        # link events temporally in the graph (adds followed_by / escalates_to edges)
        event_conf_map = {e.id: ea.confidence for e, ea in zip(events, event_alerts)}
        self.graph.link_temporal_events(events, event_conf_map)

        # basic anomaly detection on the graph
        anomalies = self.graph.basic_anomaly_detection()

        # compute forecasts for top connected people
        top_people = [p for p, _ in self.graph.top_connected_people(top_n=10)]
        forecasts = predict_for_top_people(events, {e.id: ea for e, ea in zip(events, event_alerts)}, top_people)
        
        # compute group predictions (people with multiple connections)
        group_forecasts = {}
        for person in top_people:
            person_events = [e for e in events if person in getattr(e, "people", [])]
            connected_people = set()
            for e in person_events:
                connected_people.update(getattr(e, "people", []))
            connected_people.discard(person)
            if len(connected_people) > 1:
                group = list(connected_people)[:5]
                group_forecast = predict_group_trajectory(group, events, {e.id: ea for e, ea in zip(events, event_alerts)})
                group_forecasts[person] = group_forecast
        
        # compute intervention suggestions for high-risk people
        interventions = {}
        for person in top_people:
            sugg = suggest_interventions(person, events, {e.id: ea for e, ea in zip(events, event_alerts)})
            if sugg:
                interventions[person] = sugg

        timeline = reconstruct_timeline(normalized_documents, events)

        # compute uncertainty per event using a small ensemble (heuristic + alert confidence + threshold)
        heuristic = HeuristicPredictor()
        event_uncertainty = {}
        for e, ea in zip(events, event_alerts):
            feats = {
                "current_risk": getattr(ea, "confidence", 0.0),
                "participant_count": len(getattr(e, "people", []) or []),
                "location_count": len(getattr(e, "locations", []) or []),
                "num_messages": len(getattr(e, "related_message_ids", []) or []),
                "sentiment": compute_sentiment(getattr(e, "description", "") or ""),
            }
            try:
                hpred = heuristic.predict([feats])[0]
            except Exception:
                hpred = float(getattr(ea, "confidence", 0.0))
            apred = float(getattr(ea, "confidence", 0.0))
            tpred = 1.0 if apred >= 0.7 else 0.0

            prob_vectors = [[hpred, 1 - hpred], [apred, 1 - apred], [tpred, 1 - tpred]]
            avg_probs = [sum(p[i] for p in prob_vectors) / len(prob_vectors) for i in range(2)]
            ent = entropy_from_probs(avg_probs)
            disagree = ensemble_disagreement(prob_vectors)
            unc_label = label_uncertainty(apred, ent, disagree)
            event_uncertainty[e.id] = {"entropy": ent, "disagreement": disagree, "uncertainty": unc_label, "ensemble_probs": avg_probs}

        if persist_db_path:
            from .storage import init_db, store_messages, store_events, store_event_risk

            conn = init_db(persist_db_path)
            store_messages(conn, [doc.dict() for doc in normalized_documents])
            store_events(conn, [event.dict() for event in events])
            for ea in event_alerts:
                store_event_risk(conn, ea.id, ea.level.value, ea.confidence, datetime.utcnow().isoformat())
            conn.close()

        # make operational decision per event and enqueue tasks based on policy
        decisions = {}
        enqueue_results = {}
        for e in events:
            unc = event_uncertainty.get(e.id, {})
            feats = {
                "risk_slope": 0.0,
                "participant_count": len(getattr(e, "people", []) or []),
            }
            action, reasons = decide_event(e, next((a for a in event_alerts if a.id == f"event-alert-{e.id}"), None), unc, feats, db_path=persist_db_path)
            decisions[e.id] = {"action": action, "reasons": reasons}
            # enqueue only ESCALATE or PRIORITY_ALERT
            if action in ("ESCALATE", "PRIORITY_ALERT") and persist_db_path:
                res = enqueue_for_event(persist_db_path, e.id, action, priority=10 if action == "PRIORITY_ALERT" else 7, assigned_to=None, dedup_key=f"event-{e.id}")
                enqueue_results[e.id] = res

        # optional evaluation and audit
        beliefs = generate_beliefs_from_graph(self.graph, events, {e.id: ea for e, ea in zip(events, event_alerts)})
        hypotheses = generate_competing_hypotheses(self.graph, beliefs, events)

        if persist_db_path:
            from .storage import init_db, store_messages, store_events, store_event_risk, store_belief, store_hypothesis

            conn = init_db(persist_db_path)
            store_messages(conn, [doc.dict() for doc in normalized_documents])
            store_events(conn, [event.dict() for event in events])
            for ea in event_alerts:
                store_event_risk(conn, ea.id, ea.level.value, ea.confidence, datetime.utcnow().isoformat())
            for belief in beliefs:
                store_belief(conn, belief)
            for hypothesis in hypotheses:
                store_hypothesis(conn, hypothesis)
            conn.close()

        if evaluate and ground_truth is not None:
            metrics, records = evaluate_events(events, {e.id: ea for e, ea in zip(events, event_alerts)}, ground_truth)
            if audit_db_path:
                audit_prediction(audit_db_path, event_id="*", action="evaluation_run", payload={"metrics": metrics})
            create_snapshot("pipeline_evaluation_snapshot.json", {"metrics": metrics, "records_sample": records[:20], "event_uncertainty_sample": dict(list(event_uncertainty.items())[:20])})

        return {
            "documents": [doc.dict() for doc in normalized_documents],
            "entities": [entity.dict() for entity in all_entities],
            "alerts": [alert.dict() for alert in alerts],
            "event_alerts": [ea.dict() for ea in event_alerts],
            "event_uncertainty": event_uncertainty,
            "decisions": decisions,
            "enqueue_results": enqueue_results,
            "events": [event.dict() for event in events],
            "anomalies": anomalies,
            "forecasts": forecasts,
            "group_forecasts": group_forecasts,
            "interventions": interventions,
            "timeline": timeline,
            "graph": self.graph.facts(),
            "beliefs": [belief.dict() for belief in beliefs],
            "hypotheses": hypotheses,
        }
