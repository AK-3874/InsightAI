from typing import List, Dict, Any
import random


def obfuscate_text_coded(text: str, code_map: Dict[str, str]) -> str:
    out = text
    for k, v in code_map.items():
        out = out.replace(k, v)
    return out


def split_messages(event: Dict[str, Any], split_ratio: float = 0.5) -> List[Dict[str, Any]]:
    # split an event description into multiple smaller messages
    text = event.get("description", event.get("text", ""))
    words = text.split()
    if len(words) < 4:
        return [event]
    cut = max(1, int(len(words) * split_ratio))
    a = " ".join(words[:cut])
    b = " ".join(words[cut:])
    e1 = dict(event)
    e2 = dict(event)
    e1["text"] = a
    e2["text"] = b
    return [e1, e2]


def make_sarcastic(text: str) -> str:
    # simple heuristic: append a sarcasm marker or insert mild negation
    return text + " (not)"


def adversarial_variations(documents: List[Dict[str, Any]], code_map: Dict[str, str], split_prob: float = 0.1, sarcasm_prob: float = 0.05):
    out = []
    for d in documents:
        new = dict(d)
        if random.random() < split_prob:
            parts = split_messages(d)
            out.extend(parts)
            continue
        if random.random() < sarcasm_prob:
            new_text = make_sarcastic(d.get("text", ""))
            new["text"] = new_text
        new["text"] = obfuscate_text_coded(new.get("text", ""), code_map)
        out.append(new)
    return out
