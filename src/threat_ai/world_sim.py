import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .models import Document, SourceType

PERSON_PREFIXES = ["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Jamie", "Drew", "Robin", "Quinn"]
LOCATIONS = ["warehouse", "airport", "hotel", "office", "dock", "factory", "camp", "alley", "parking lot", "station", "mall", "train yard"]
ORGANIZATIONS = ["North Group", "East Logistics", "Silver Team", "Delta Corp", "Redfire", "Shadow Ops", "Blue Network", "Green Alliance"]
EVENT_TEMPLATES = [
    "Meet at {location} at {time}.",
    "Bring the package to {location}.",
    "We need to transfer the funds through {organization}.",
    "The meeting at {location} is confirmed.",
    "Coordinate the next drop with {person}.",
    "If you do not comply, there will be consequences.",
    "This needs to happen tonight at {location}.",
    "The operation begins at {time}.",
    "Use the side entrance at {location}.",
    "The team gathers near {location}.",
]

ESCALATION_TEMPLATES = [
    "If you do not comply, you will regret it.",
    "This will get ugly if it continues.",
    "We are moving forward regardless.",
    "The threat level is increasing.",
    "There is no second warning.",
    "Keep this quiet or there will be trouble.",
]


def _make_person(index: int) -> str:
    return f"{random.choice(PERSON_PREFIXES)}_{index}"


def _make_time(start: datetime, minutes: int) -> datetime:
    return start + timedelta(minutes=minutes)


def _assign_participants(num_people: int, people: List[str], count: int = 3) -> List[str]:
    return random.sample(people, min(count, len(people)))


def build_world(num_people: int = 1000, num_events: int = 2000, escalations: float = 0.15) -> Tuple[List[Document], Dict[str, Dict[str, object]]]:
    people = [_make_person(i) for i in range(1, num_people + 1)]
    documents: List[Document] = []
    ground_truth: Dict[str, Dict[str, object]] = {}
    current_time = datetime.utcnow() - timedelta(days=30)
    event_counter = 1

    for i in range(num_events):
        message_time = _make_time(current_time, random.randint(0, 10))
        current_time = message_time
        location = random.choice(LOCATIONS)
        organization = random.choice(ORGANIZATIONS)
        participants = _assign_participants(num_people, people, count=random.randint(2, 6))
        person_ref = random.choice(participants)
        template = random.choice(EVENT_TEMPLATES)
        text = template.format(location=location, time=message_time.strftime("%H:%M"), organization=organization, person=person_ref)
        if random.random() < escalations:
            text += " " + random.choice(ESCALATION_TEMPLATES)
            escalated = True
            escalation_time = message_time + timedelta(minutes=random.randint(10, 240))
        else:
            escalated = False
            escalation_time = None

        doc = Document(
            id=f"doc-{event_counter}",
            source_type=SourceType.EMAIL,
            source_name="sim",
            timestamp=message_time,
            text=text,
            metadata={"participants": participants, "location": location, "organization": organization},
        )
        documents.append(doc)
        ground_truth[doc.id] = {
            "escalated": escalated,
            "escalation_time": escalation_time,
            "participants": participants,
            "location": location,
        }
        event_counter += 1
        current_time += timedelta(minutes=random.randint(1, 60))

    return documents, ground_truth


def generate_missing_message_scenario(documents: List[Document], missing_rate: float = 0.3) -> List[Document]:
    output = []
    for doc in documents:
        if random.random() < missing_rate:
            continue
        output.append(doc)
    return output


def reassign_entities(documents: List[Document], change_rate: float = 0.2) -> List[Document]:
    people = set(p for doc in documents for p in doc.metadata.get("participants", []))
    mapping = {}
    for person in people:
        if random.random() < change_rate:
            mapping[person] = person + "_alias"
    updated = []
    for doc in documents:
        text = doc.text
        for old, new in mapping.items():
            text = text.replace(old, new)
        new_doc = Document(**{**doc.dict(), "text": text, "metadata": {**doc.metadata}})
        updated.append(new_doc)
    return updated


@dataclass
class CityPerson:
    name: str
    home: str
    workplace: str
    organization: str
    friends: List[str] = field(default_factory=list)
    family: List[str] = field(default_factory=list)
    business_contacts: List[str] = field(default_factory=list)
    travel_pattern: List[str] = field(default_factory=list)


def build_synthetic_city(num_people: int = 1000, num_homes: int = 50, num_workplaces: int = 50, num_organizations: int = 20) -> Dict[str, Any]:
    homes = [f"Neighborhood {chr(65 + i)}" for i in range(min(num_homes, 26))]
    workplaces = [f"Site {i + 1}" for i in range(num_workplaces)]
    organizations = ORGANIZATIONS[: min(num_organizations, len(ORGANIZATIONS))]

    people = []
    for i in range(1, num_people + 1):
        name = _make_person(i)
        home = random.choice(homes)
        workplace = random.choice(workplaces)
        organization = random.choice(organizations)
        person = CityPerson(name=name, home=home, workplace=workplace, organization=organization)
        people.append(person)

    # build families by home clusters
    by_home = {}
    for person in people:
        by_home.setdefault(person.home, []).append(person.name)
    for home, residents in by_home.items():
        random.shuffle(residents)
        while residents:
            max_size = min(5, len(residents))
            size = random.randint(1, max_size) if max_size == 1 else random.randint(2, max_size)
            family = [residents.pop() for _ in range(size)]
            for member in family:
                p = next(person for person in people if person.name == member)
                p.family.extend([x for x in family if x != member])

    # friendships and business contacts
    for person in people:
        friends = set(random.sample([p.name for p in people if p.name != person.name], min(8, len(people) - 1)))
        businesses = set(random.sample([p.name for p in people if p.name != person.name], min(6, len(people) - 1)))
        # prefer same home or workplace
        same_home = [p.name for p in people if p.home == person.home and p.name != person.name]
        same_work = [p.name for p in people if p.workplace == person.workplace and p.name != person.name]
        person.friends = list(set(random.sample(same_home + same_work, min(5, len(same_home + same_work))) + list(friends)))[:8]
        person.business_contacts = list(set(random.sample(same_work, min(4, len(same_work))) + list(businesses)))[:6]
        commutes = [person.home, person.workplace, random.choice(LOCATIONS)]
        person.travel_pattern = commutes[:3]

    return {
        "people": people,
        "homes": homes,
        "workplaces": workplaces,
        "organizations": organizations,
    }


def simulate_city_events(city: Dict[str, Any], days: int = 7, events_per_day: int = 100, escalation_rate: float = 0.12) -> Tuple[List[Document], Dict[str, Dict[str, object]]]:
    people = city["people"]
    locations = LOCATIONS + city["workplaces"]
    organizations = city["organizations"]
    documents: List[Document] = []
    ground_truth: Dict[str, Dict[str, object]] = {}
    start_time = datetime.utcnow() - timedelta(days=days)
    event_counter = 1

    for day in range(days):
        for _ in range(events_per_day):
            base_time = start_time + timedelta(days=day, minutes=random.randint(0, 1439))
            participants = random.sample([p.name for p in people], random.randint(2, 5))
            location = random.choice(locations)
            organization = random.choice(organizations)
            template = random.choice(EVENT_TEMPLATES)
            text = template.format(
                location=location,
                time=base_time.strftime("%H:%M"),
                organization=organization,
                person=random.choice(participants),
            )
            if random.random() < escalation_rate:
                text += " " + random.choice(ESCALATION_TEMPLATES)
                escalated = True
            else:
                escalated = False
            doc = Document(
                id=f"doc-{event_counter}",
                source_type=random.choice(list(SourceType)),
                source_name=organization,
                timestamp=base_time,
                text=text,
                metadata={
                    "participants": participants,
                    "location": location,
                    "organization": organization,
                    "scenario": None,
                },
            )
            documents.append(doc)
            ground_truth[doc.id] = {
                "escalated": escalated,
                "escalation_time": base_time + timedelta(hours=random.randint(1, 8)) if escalated else None,
                "participants": participants,
                "location": location,
                "scenario": None,
                "timestamp": base_time,
            }
            event_counter += 1

    return documents, ground_truth


def inject_hidden_scenarios(documents: List[Document], ground_truth: Dict[str, Dict[str, object]], city: Dict[str, Any], scenario_types: List[str]) -> None:
    people = [p.name for p in city["people"]]
    for scenario in scenario_types:
        cluster = random.sample(people, min(8, len(people)))
        scenario_start = min(doc.timestamp for doc in documents) + timedelta(days=random.randint(0, max(1, len(documents) // 500)))
        for i in range(4):
            timestamp = scenario_start + timedelta(hours=3 * i)
            participants = random.sample(cluster, random.randint(3, min(6, len(cluster))))
            location = random.choice(LOCATIONS)
            organization = random.choice(city["organizations"])
            if scenario == "fraud_ring":
                text = f"Transfer funds through {organization} and verify account details with {participants[0]}."
            elif scenario == "harassment_campaign":
                text = f"{participants[0]} is the target; keep up the pressure and do not stop." 
            elif scenario == "coordinated_activity":
                text = f"Meet at {location} with the team at {timestamp.strftime('%H:%M')} to coordinate the next action."
            else:
                text = f"Escalate the plan at {location} tonight; failure is not an option."
            text += " " + random.choice(ESCALATION_TEMPLATES) if i >= 2 else ""
            doc = Document(
                id=f"hidden-{scenario}-{i + 1}",
                source_type=random.choice(list(SourceType)),
                source_name=organization,
                timestamp=timestamp,
                text=text,
                metadata={
                    "participants": participants,
                    "location": location,
                    "organization": organization,
                    "scenario": scenario,
                },
            )
            documents.append(doc)
            ground_truth[doc.id] = {
                "escalated": True,
                "escalation_time": timestamp + timedelta(hours=(4 - i)),
                "participants": participants,
                "location": location,
                "scenario": scenario,
                "timestamp": timestamp,
            }


def build_city_world(
    num_people: int = 1000,
    days: int = 7,
    events_per_day: int = 100,
    hidden_scenarios: Optional[List[str]] = None,
) -> Tuple[List[Document], Dict[str, Dict[str, object]], Dict[str, Any]]:
    city = build_synthetic_city(num_people=num_people)
    documents, ground_truth = simulate_city_events(city, days=days, events_per_day=events_per_day)
    if hidden_scenarios is None:
        hidden_scenarios = ["fraud_ring", "harassment_campaign", "coordinated_activity", "escalating_threat"]
    inject_hidden_scenarios(documents, ground_truth, city, hidden_scenarios)
    documents = sorted(documents, key=lambda d: d.timestamp or datetime.min)
    return documents, ground_truth, city
