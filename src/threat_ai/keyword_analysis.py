"""Keyword matching analysis for risk detector."""

from typing import Dict, List, Tuple

# From risk.py
THREAT_KEYWORDS = [
    "kill", "explode", "bomb", "shoot", "attack", "weapon", "raid", "threat", "hostage", "pay the price", "regret it",
]

HARASSMENT_KEYWORDS = [
    "idiot", "stupid", "shut up", "worthless", "loser", "harass", "insult", "hate",
]

FRAUD_KEYWORDS = [
    "bank details", "credit card", "account number", "social security", "wire transfer", "send me money", "password", "verify your identity",
]

SUSPICIOUS_MEETING_KEYWORDS = [
    "meet behind", "midnight", "warehouse", "alley", "after hours", "private meeting", "secret location",
]

ORGANIZED_ACTIVITY_KEYWORDS = [
    "coordinate", "organized", "team up", "group session", "plan together", "recruit",
]

KEYWORD_GROUPS = {
    "threat": THREAT_KEYWORDS,
    "harassment": HARASSMENT_KEYWORDS,
    "fraud": FRAUD_KEYWORDS,
    "suspicious_meeting": SUSPICIOUS_MEETING_KEYWORDS,
    "organized_activity": ORGANIZED_ACTIVITY_KEYWORDS,
}

# Injected scenario examples
SCENARIO_TEXTS = {
    "fraud_ring": "Transfer funds through East Logistics and verify account details with Jordan_132",
    "harassment_campaign": "Drew_120 is the target; keep up the pressure and do not stop.",
    "coordinated_activity": "Meet at airport with the team at 21:46 to coordinate the next action.",
    "escalating_threat": "Escalate the plan at camp tonight; failure is not an option.",
}


def analyze_keyword_coverage() -> Dict[str, Dict[str, any]]:
    """Check how many injected scenarios match risk detector keywords."""
    results = {}

    for scenario_name, text in SCENARIO_TEXTS.items():
        lower_text = text.lower()
        hits = {}
        misses = {}

        for keyword_group, keywords in KEYWORD_GROUPS.items():
            group_hits = [kw for kw in keywords if kw in lower_text]
            if group_hits:
                hits[keyword_group] = group_hits
            else:
                misses[keyword_group] = keywords

        results[scenario_name] = {
            "text": text,
            "hits": hits,
            "misses": misses,
            "matched": bool(hits),
        }

    return results


def print_keyword_analysis():
    """Print detailed keyword matching for each injected scenario."""
    analysis = analyze_keyword_coverage()

    print("=== KEYWORD MATCHING ANALYSIS ===\n")

    for scenario, data in analysis.items():
        print(f"{scenario.upper()}")
        print(f"  Text: {data['text']}")
        print(f"  Matched: {data['matched']}")

        if data["hits"]:
            print(f"  ✓ Hits:")
            for group, keywords in data["hits"].items():
                print(f"    - {group}: {keywords}")
        else:
            print(f"  ✗ No keyword matches")

        if data["misses"]:
            print(f"  ✗ Missed by groups:")
            for group, keywords in data["misses"].items():
                print(f"    - {group}: {keywords[:2]}...")  # Just show first 2

        print()

    # Summary
    total = len(SCENARIO_TEXTS)
    matched = sum(1 for data in analysis.values() if data["matched"])
    print(f"Summary: {matched}/{total} scenarios matched risk detector keywords")


if __name__ == "__main__":
    print_keyword_analysis()
