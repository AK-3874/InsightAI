import json
import random
from typing import List

from .models import RiskType

LABEL_TEMPLATES = {
    "normal": [
        "Let's grab lunch tomorrow.",
        "I'll send the agenda before the meeting.",
        "Please review the document and share feedback.",
        "Can we reschedule for next week?",
        "The event is confirmed for Thursday.",
        "Thank you for your help today.",
        "The report has been uploaded.",
        "I will call you later this evening.",
        "Let's discuss this during tomorrow's meeting.",
        "The package arrived safely.",
        "Can you check the latest version?",
        "I'll be working remotely today.",
        "Please let me know if you need anything.",
        "The presentation went well.",
        "Looking forward to seeing everyone next week.",
        "Let's finalize the budget tomorrow.",
        "The project is progressing as planned.",
        "Thanks for sending the information.",
        "I'll review it after lunch.",
        "The client approved the proposal."
    ],

    "threat": [
        "If you don't comply, there will be consequences.",
        "I will make sure you face the results.",
        "Do not ignore this or you will regret it.",
        "This needs to be handled immediately or there will be trouble.",
        "You have been warned.",
        "Keep talking and you'll see what happens.",
        "Nobody will protect you.",
        "You won't like the outcome.",
        "I know where you live.",
        "This is your final warning.",
        "You better stay out of my way.",
        "Things will get ugly if you continue.",
        "You are making a dangerous mistake.",
        "Watch your back.",
        "I can make your life difficult.",
        "You should be afraid of what comes next.",
        "There will be a price to pay.",
        "You'll wish you had listened.",
        "Don't test me.",
        "You're going to regret this."
    ],

    "harassment": [
        "Nobody wants you here.",
        "You are completely useless.",
        "Everyone is tired of dealing with you.",
        "You embarrass yourself every day.",
        "Why are you always making mistakes?",
        "Nobody takes you seriously.",
        "You should quit.",
        "You're a burden to this team.",
        "Stop talking, nobody cares.",
        "You never do anything right.",
        "You're pathetic.",
        "Everyone laughs at you.",
        "You're the reason this failed.",
        "Nobody likes working with you.",
        "You keep ruining everything.",
        "You make every meeting worse.",
        "You're wasting everyone's time.",
        "Nobody asked for your opinion.",
        "You don't belong here.",
        "You're hopeless."
    ],

    "fraud": [
        "Please share your bank details for reimbursement.",
        "Verify your identity by providing your password.",
        "Transfer the funds as soon as possible.",
        "Click this link to secure your account.",
        "Your account has been suspended. Login immediately.",
        "We need your credit card number.",
        "Send your security code for verification.",
        "Provide your banking credentials.",
        "Urgent payment required today.",
        "Wire the money to this account.",
        "Confirm your password immediately.",
        "Your account will be locked unless you act now.",
        "Update your payment information.",
        "Send the authentication code you received.",
        "Claim your reward by sharing your details.",
        "A processing fee is required.",
        "Verify your identity through this form.",
        "Your account has unusual activity.",
        "Complete the payment using the attached instructions.",
        "Share your PIN for confirmation."
    ],

    "suspicious_meeting": [
        "Meet behind the building at midnight.",
        "Let's have a private meeting after hours.",
        "The group will gather at the secret location.",
        "Come to the warehouse for a quick discussion.",
        "Use the side entrance after dark.",
        "Nobody should know about this meeting.",
        "Meet where the cameras can't see us.",
        "Leave your phone at home.",
        "We'll discuss the details in person.",
        "Meet me near the abandoned lot.",
        "Come alone.",
        "Use the back alley entrance.",
        "Don't tell anyone where you're going.",
        "Meet after everyone leaves.",
        "The location will be shared later.",
        "Avoid the main entrance.",
        "Wait until midnight before arriving.",
        "Use a different route.",
        "The meeting is off the record.",
        "Keep this conversation private.",
        "See you at the warehouse.",
        "Let's talk in private."
    ],

    "self_harm": [
        "I feel like I can't keep going.",
        "Sometimes I think it would be easier if I disappeared.",
        "Life feels too hard right now.",
        "I'm struggling more than usual today.",
        "I don't see a way forward.",
        "Everything feels pointless.",
        "I feel completely exhausted emotionally.",
        "I don't know how much longer I can do this.",
        "Nothing seems to matter anymore.",
        "I feel empty inside.",
        "I wish I could stop feeling this way.",
        "Every day feels harder.",
        "I'm losing hope.",
        "I feel trapped.",
        "I don't want to wake up tomorrow.",
        "I can't handle the pressure anymore.",
        "I'm overwhelmed by everything.",
        "Nobody would notice if I disappeared.",
        "I feel broken.",
        "I don't know what to do anymore."
    ],

    "organized_activity": [
        "Coordinate the next team session for Friday.",
        "We should plan the group activity together.",
        "Recruit more people to help with the event.",
        "Organize a session for the full team next week.",
        "Assign responsibilities to each member.",
        "Let's schedule the next planning meeting.",
        "Build a list of volunteers.",
        "We need better coordination.",
        "Expand the group before launch.",
        "Prepare the logistics for Saturday.",
        "Gather the team leaders.",
        "Invite additional participants.",
        "Create a communication channel.",
        "Plan the next phase together.",
        "Organize transportation for the group.",
        "Prepare materials for everyone.",
        "Coordinate efforts across teams.",
        "Let's increase participation.",
        "Arrange the upcoming workshop.",
        "Manage the event schedule."
    ],
    "ambiguous": [
    "We should talk later.",
    "Send it when you can.",
    "Meet me after work.",
    "I need to see you.",
    "Let's handle this privately.",
    "I'll explain in person.",
    "Don't share this message."
    ]
}

EXTRA_PHRASES = [
"Tomorrow at 10 AM.",
"Send details by end of day.",
"Use the back entrance.",
"Location is confidential.",
"Bring what I asked for.",
"Do not mention this to anyone."
]

ADVERSARIAL_PHRASES = [
"Nice shop you run here, shame if something happens to it.",
"It would be unfortunate if things escalated.",
"I would hate for accidents to occur around your place.",
"You seem like a smart person, avoid problems and cooperate.",
"Nothing needs to get messy if you understand me.",
"People who refuse usually regret it later."
]

def build_synthetic_dataset(size: int = 1200) -> List[dict]:
    if size <= 0:
        return []

    labels = list(LABEL_TEMPLATES.keys())
    dataset = []
    next_id = 1

    while len(dataset) < size:
        label = random.choice(labels)
        template = random.choice(LABEL_TEMPLATES[label])
        if random.random() < 0.25:
            template = f"{template} {random.choice(EXTRA_PHRASES)}"

        dataset.append(
            {
                "id": f"sample-{next_id}",
                "message": template,
                "label": label,
                "risk_type": (
                    RiskType[label.upper()].value 
                    if label != "normal" and label in ["threat", "harassment", "fraud", "suspicious_meeting", "self_harm", "organized_activity"]
                    else RiskType.NORMAL.value if label == "normal"
                    else RiskType.OTHER.value
                ),
            }
        )
        next_id += 1

    return dataset


def write_dataset(path: str, size: int = 1200) -> None:
    data = build_synthetic_dataset(size=size)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


if __name__ == "__main__":
    write_dataset("synthetic_risk_dataset.json", size=600)
