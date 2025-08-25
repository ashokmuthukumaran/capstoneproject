# tools_gemini.py
from __future__ import annotations
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

from gemini_client import GeminiClient
from prompts import (
    SYSTEM_CLASSIFIER,
    SYSTEM_BUG_ANALYZER,
    SYSTEM_FEATURE_EXTRACTOR,
    SYSTEM_TICKET_CREATOR,
    SYSTEM_CRITIC,
)

# ---------- Data Model ----------

@dataclass
class Ticket:
    ticket_id: str
    source_id: str
    source_type: str
    category: str
    priority: str
    title: str
    technical_details: str
    created_at: str
    confidence: float
    link_back: str

# ---------- Deterministic fallback rules ----------

BUG_TRIGGERS = [
    "crash","error","exception","freeze","not working","can't login","cannot login",
    "login issue","stopped working","data loss","sync issue","sync not","fails to","bug"
]
FEATURE_TRIGGERS = ["please add","would love","feature request","missing","could you","add support","add dark mode"]
PRAISE_TRIGGERS = ["love","amazing","perfect","best","smooth","works great"]
COMPLAINT_TRIGGERS = ["slow","lag","expensive","poor","bad","annoying","ads","too many ads","pricey","support is"]
SPAM_TRIGGERS = ["http","www","visit","free","money","channel","asdf","subscribe","coins","discount code"]

def _contains_any(text: str, needles: List[str]) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

def fallback_classify(text: str, rating: Optional[float] = None) -> Tuple[str, float, str]:
    if _contains_any(text, SPAM_TRIGGERS):
        return "Spam", 0.95, "spam trigger"
    if _contains_any(text, BUG_TRIGGERS):
        conf = 0.9 if ("crash" in text.lower() or "data loss" in text.lower()) else 0.75
        return "Bug", conf, "bug trigger"
    if _contains_any(text, FEATURE_TRIGGERS):
        return "Feature Request", 0.8, "feature trigger"
    if _contains_any(text, PRAISE_TRIGGERS):
        return "Praise", 0.7, "praise trigger"
    if _contains_any(text, COMPLAINT_TRIGGERS):
        return "Complaint", 0.7, "complaint trigger"
    if rating is not None:
        if rating <= 2:
            return "Complaint", 0.55, "rating fallback"
        if rating >= 4:
            return "Praise", 0.55, "rating fallback"
    return "Other", 0.4, "default fallback"

def fallback_bug_analysis(text: str, platform: str, app_version: str) -> Tuple[str, str, str]:
    sev = "High" if ("crash" in text.lower() or "data loss" in text.lower() or "can't login" in text.lower() or "cannot login" in text.lower()) else "Medium"
    details = f"Platform={platform}; Version={app_version}; Summary={text[:140]}"
    return sev, details, "rule-based severity"

def fallback_feature(text: str) -> Tuple[str, str, str]:
    low = ["widget","export","multiple accounts"]
    high = ["dark mode","calendar","integration"]
    tl = text.lower()
    if any(h in tl for h in high):
        return "High", f"Feature: {text[:140]}", "high-impact trigger"
    if any(l in tl for l in low):
        return "Medium", f"Feature: {text[:140]}", "medium-impact trigger"
    return "Low", f"Feature: {text[:140]}", "low-impact default"

# ---------- CSV Utility ----------

def read_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ---------- Gemini-powered tools with fallback ----------

def classify_with_gemini(g: GeminiClient, text: str, rating: Optional[float] = None) -> Tuple[str, float, str]:
    payload = {"text": text, "rating": rating}
    if g.enabled:
        out = g.ask_json(SYSTEM_CLASSIFIER, json.dumps(payload))
        if isinstance(out, dict) and "category" in out:
            cat = out.get("category", "Other")
            conf = float(out.get("confidence", 0.6))
            rationale = out.get("brief_rationale", "")
            return cat, conf, rationale
    return fallback_classify(text, rating)

def analyze_bug_with_gemini(g: GeminiClient, text: str, platform: str, app_version: str) -> Tuple[str, str, str]:
    payload = {"text": text, "platform": platform, "version": app_version}
    if g.enabled:
        out = g.ask_json(SYSTEM_BUG_ANALYZER, json.dumps(payload))
        if isinstance(out, dict) and "severity" in out:
            return out.get("severity", "Medium"), out.get("technical_details", ""), out.get("brief_rationale","")
    return fallback_bug_analysis(text, platform, app_version)

def extract_feature_with_gemini(g: GeminiClient, text: str) -> Tuple[str, str, str]:
    payload = {"text": text}
    if g.enabled:
        out = g.ask_json(SYSTEM_FEATURE_EXTRACTOR, json.dumps(payload))
        if isinstance(out, dict) and "impact" in out:
            return out.get("impact", "Low"), out.get("details", f"Feature: {text[:140]}"), out.get("suggested_title","Feature")
    return fallback_feature(text)

def compose_ticket_with_gemini(g: GeminiClient, title_hint: str, body_hint: str) -> Tuple[str, str]:
    payload = {"title_hint": title_hint[:120], "body_hint": body_hint[:500]}
    if g.enabled:
        out = g.ask_json(SYSTEM_TICKET_CREATOR, json.dumps(payload))
        if isinstance(out, dict) and "title" in out and "body" in out:
            return out["title"][:80], out["body"]
    # fallback: just trim
    return title_hint[:80], body_hint[:400]

def critic_with_gemini(g: GeminiClient, ticket: Dict[str, Any]) -> Dict[str, Any]:
    if g.enabled:
        out = g.ask_json(SYSTEM_CRITIC, json.dumps(ticket))
        if isinstance(out, dict):
            if out.get("ok") is True:
                return ticket
            # merge corrections
            corrected = ticket.copy()
            for k, v in out.items():
                if k != "ok":
                    corrected[k] = v
            return corrected
    # fallback: minimal sanity
    corrected = ticket.copy()
    if corrected.get("category") in ["Spam","Praise"] and corrected.get("priority") in ["High","Critical"]:
        corrected["priority"] = "Low"
    return corrected

# ---------- Ticket creation & metrics ----------

def create_ticket(feedback_id: str, source_type: str, category: str,
                  priority: str, title: str, technical_details: str,
                  confidence: float, link_back: str) -> Ticket:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return Ticket(
        ticket_id=f"T{feedback_id}",
        source_id=str(feedback_id),
        source_type=source_type,
        category=category,
        priority=priority,
        title=title,
        technical_details=technical_details,
        created_at=now,
        confidence=confidence,
        link_back=link_back or ""
    )

def compute_metrics(processing_log: pd.DataFrame,
                    expected: pd.DataFrame | None = None) -> pd.DataFrame:
    metrics = {}
    metrics["total_items"] = len(processing_log)
    for k in ["Bug","Feature Request","Praise","Complaint","Spam","Other"]:
        metrics[k.lower().replace(" ", "_")] = int((processing_log["category"] == k).sum())
    metrics["avg_confidence"] = float(processing_log["confidence"].mean() if len(processing_log) else 0.0)
    if expected is not None and "category" in expected.columns:
        merged = processing_log.merge(expected, on="source_id", suffixes=("", "_expected"))
        if "category_expected" in merged.columns:
            acc = (merged["category"] == merged["category_expected"]).mean() if len(merged) else 0.0
            metrics["category_accuracy"] = float(acc)
    return pd.DataFrame([metrics])
