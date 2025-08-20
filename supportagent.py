import pandas as pd
import re
import random
from datetime import datetime

# -------------------------
# Agent 1: CSV Reader Agent
# -------------------------
class CSVReaderAgent:
    def __init__(self, filepath):
        self.filepath = filepath
    
    def read_data(self):
        return pd.read_csv(self.filepath)


# --------------------------------
# Agent 2: Feedback Classifier Agent
# --------------------------------
class FeedbackClassifierAgent:
    def classify(self, text, rating=None):
        text_lower = text.lower()
        if any(word in text_lower for word in ["crash", "bug", "error", "freeze", "not working", "login"]):
            return "Bug"
        elif any(word in text_lower for word in ["please add", "would love", "feature request", "missing", "could you"]):
            return "Feature Request"
        elif any(word in text_lower for word in ["love", "amazing", "perfect", "best", "smooth"]):
            return "Praise"
        elif any(word in text_lower for word in ["slow", "expensive", "poor", "bad", "annoying", "ads"]):
            return "Complaint"
        elif any(word in text_lower for word in ["http", "visit", "free", "money", "channel", "asdf"]):
            return "Spam"
        else:
            # fallback using rating
            if rating and rating <= 2:
                return "Complaint"
            elif rating and rating >= 4:
                return "Praise"
            return "Other"


# -----------------------------
# Agent 3: Bug Analysis Agent
# -----------------------------
class BugAnalysisAgent:
    def analyze(self, text, platform, version):
        severity = "High" if any(word in text.lower() for word in ["crash", "data loss"]) else "Medium"
        technical_details = f"Platform: {platform}, App Version: {version}, Issue: {text[:80]}"
        return severity, technical_details


# -------------------------------
# Agent 4: Feature Extractor Agent
# -------------------------------
class FeatureExtractorAgent:
    def extract(self, text):
        impact = random.choice(["High", "Medium", "Low"])  # mock scoring
        return impact, f"Feature Request: {text[:100]}"


# ------------------------------
# Agent 5: Ticket Creator Agent
# ------------------------------
class TicketCreatorAgent:
    def create_ticket(self, feedback_id, source_type, category, priority, details, title):
        return {
            "ticket_id": f"T{feedback_id}",
            "source_id": feedback_id,
            "source_type": source_type,
            "category": category,
            "priority": priority,
            "technical_details": details,
            "title": title,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# ------------------------------
# Agent 6: Quality Critic Agent
# ------------------------------
class QualityCriticAgent:
    def review(self, ticket):
        # Ensure mandatory fields
        required = ["ticket_id", "category", "priority", "title"]
        for field in required:
            if not ticket.get(field):
                ticket["priority"] = "Needs Review"
        return ticket
