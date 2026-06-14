#!/usr/bin/env python
"""Quick end-to-end test of the threat detection pipeline."""

from datetime import datetime, timedelta
from src.threat_ai.pipeline import ThreatPipeline
from src.threat_ai.models import Document, SourceType
from src.threat_ai.sample_data import build_synthetic_dataset
from src.threat_ai.storage import init_db, store_messages, store_events, store_event_risk

# Generate synthetic documents
print("Generating synthetic dataset...")
data = build_synthetic_dataset(size=50)
docs = [
    Document(
        id=d["id"],
        source_type=SourceType.CHAT,
        source_name="test_chat",
        text=d["message"],
        timestamp=datetime.utcnow() - timedelta(days=i % 7)
    )
    for i, d in enumerate(data)
]

# Run the threat detection pipeline
print("Running threat detection pipeline...")
tp = ThreatPipeline()
output = tp.run(docs)

print(f"\nPipeline Output Summary:")
print(f"  Documents processed: {len(output['documents'])}")
print(f"  Entities extracted: {len(output['entities'])}")
print(f"  Message-level alerts: {len(output['alerts'])}")
print(f"  Events formed: {len(output['events'])}")
print(f"  Event-level alerts: {len(output['event_alerts'])}")
print(f"  Anomalies detected: {len(output['anomalies'])}")
print(f"  Individual forecasts: {len(output['forecasts'])}")
print(f"  Group forecasts: {len(output['group_forecasts'])}")
print(f"  Intervention suggestions: {len(output['interventions'])}")
print(f"  Timeline entries: {len(output['timeline'])}")
print(f"  Graph facts: {len(output['graph'])}")

# Show sample forecast
if output['forecasts']:
    first_person = list(output['forecasts'].keys())[0]
    forecast = output['forecasts'][first_person]
    print(f"\nSample Forecast for {first_person}:")
    print(f"  Current risk: {forecast['forecast']['current']:.3f}")
    print(f"  1-day forecast: {forecast['forecast']['forecast'].get(1, 0):.3f}")
    print(f"  7-day forecast: {forecast['forecast']['forecast'].get(7, 0):.3f}")
    print(f"  Confidence: {forecast['forecast']['confidence']:.3f}")

# Show sample intervention
if output['interventions']:
    first_person = list(output['interventions'].keys())[0]
    interventions = output['interventions'][first_person]
    print(f"\nSample Interventions for {first_person}:")
    for i, intervention in enumerate(interventions[:2]):
        print(f"  {i+1}. [{intervention['priority'].upper()}] {intervention['action']}")
        print(f"     Reason: {intervention['reason']}")

print("\nOK - End-to-end test completed successfully!")
