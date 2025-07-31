# app/parse_query.py

import re
from typing import Dict, List

def parse_query(query: str) -> Dict:
    """
    Parses a health insurance-related query to extract semantic tags.

    Args:
        query (str): User input question or phrase.

    Returns:
        dict: {
            "original_query": str,
            "tags": List[str],
            "has_medical": bool,
            "has_benefit": bool
        }
    """
    query = query.strip().lower()

    keyword_map = {
        "surgery": ["surgery", "operation", "procedure"],
        "maternity": ["maternity", "pregnancy", "childbirth"],
        "hospital": ["hospital", "facility", "room rent", "icu"],
        "waiting_period": ["waiting period", "eligibility", "initial wait"],
        "pre_existing": ["pre-existing", "ped", "pre existing"],
        "discount": ["ncd", "discount", "no claim bonus"],
        "ayush": ["ayurveda", "homeopathy", "ayush", "unani"],
        "organ_donor": ["organ donor", "transplant", "donation"],
        "grace_period": ["grace period", "payment grace", "premium grace", "due date", "late payment"]
    }

    found_tags: List[str] = []

    for tag, terms in keyword_map.items():
        if any(re.search(rf"\b{re.escape(term)}\b", query) for term in terms):
            found_tags.append(tag)

    return {
        "original_query": query,
        "tags": found_tags,
        "has_medical": any(tag in found_tags for tag in {"surgery", "maternity", "pre_existing", "organ_donor"}),
        "has_benefit": any(tag in found_tags for tag in {"discount", "ayush"})
    }
