    # autogen_pipeline.py
"""
AutoGen-based orchestration for the Gemini-powered feedback pipeline.
- AutoGen handles agent roles & conversation skeleton.
- Gemini (google-generativeai) does the core reasoning (classification, extraction, critique).
- If GEMINI_API_KEY is missing, deterministic fallbacks are used.
"""
from __future__ import annotations
import os
import json
import pandas as pd
from typing import Dict, Any

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

from gemini_client import GeminiClient
from tools_gemini import (
    read_csv_file,
    classify_with_gemini,
    analyze_bug_with_gemini,
    extract_feature_with_gemini,
    compose_ticket_with_gemini,
    critic_with_gemini,
    create_ticket,
    compute_metrics,
)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def main(config_path: str = "config.json"):
    cfg = load_config(config_path if os.path.exists("config.json") else "config.example.json")
    gemini_cfg = cfg.get("gemini", {})
    g = GeminiClient(
        model=gemini_cfg.get("model", "gemini-1.5-flash"),
        temperature=float(gemini_cfg.get("temperature", 0.2)),
        top_p=float(gemini_cfg.get("top_p", 0.95)),
        top_k=int(gemini_cfg.get("top_k", 40)),
    )

    # --- Define agents (LLM disabled internally; we call Gemini explicitly via tools) ---
    reader = AssistantAgent(name="CSVReaderAgent", llm_config=False)
    classifier = AssistantAgent(name="FeedbackClassifierAgent", llm_config=False)
    bugger = AssistantAgent(name="BugAnalysisAgent", llm_config=False)
    featurex = AssistantAgent(name="FeatureExtractorAgent", llm_config=False)
    ticketor = AssistantAgent(name="TicketCreatorAgent", llm_config=False)
    critic = AssistantAgent(name="QualityCriticAgent", llm_config=False)
    logger = AssistantAgent(name="LoggerAgent", llm_config=False)

    user = UserProxyAgent(
        name="Orchestrator",
        human_input_mode="NEVER",
        code_execution_config=False,
        system_message="Coordinates the pipeline deterministically; Gemini is called from tools."
    )

    groupchat = GroupChat(agents=[user, reader, classifier, bugger, featurex, ticketor, critic, logger], messages=[])
    manager = GroupChatManager(groupchat=groupchat, llm_config=False)

    files = cfg["files"]
    reviews_csv = files.get("app_store_reviews", "app_store_reviews.csv")
    support_csv = files.get("support_emails", "support_emails.csv")
    expected_csv = files.get("expected_classifications", "expected_classifications.csv")

    if os.path.exists(reviews_csv):
        df_reviews = read_csv_file(reviews_csv)
    else:
        df_reviews = pd.DataFrame()
        
    if support_csv and os.path.exists(support_csv):
        df_support = read_csv_file(support_csv)
    else:
        df_support = pd.DataFrame()

    def row_to_ticket(row: pd.Series, source_type: str):
        """
        Converts a pandas Series representing a user review or feedback row into a structured ticket object.
        Args:
            row (pd.Series): The input row containing review or feedback data.
            source_type (str): The source type identifier (e.g., "AppStore", "Email").
        Returns:
            Ticket: A ticket object created from the row data, containing fields such as ID, source type, category,
                    priority, title, body, confidence score, and link back to the original review.
        The function performs the following steps:
            - Extracts relevant text and metadata from the row.
            - Classifies the review into categories (Bug, Feature Request, Complaint, Praise, Spam) using Gemini.
            - For Bugs and Feature Requests, further analyzes severity/impact and extracts technical or feature details.
            - Maps severity or impact to ticket priority.
            - Composes a ticket title and body using Gemini.
            - Returns a ticket object with all relevant information.
        """
        text = row.get("review_text") or row.get("body") or ""
        rating = row.get("rating", None)
        category, conf, _ = classify_with_gemini(g, text, rating)

        priority = "Low"
        title_hint = text[:72] or f"{category} item"
        details = f"Summary={text[:240]}"

        platform = row.get("platform", row.get("device", "Unknown"))
        version = row.get("app_version", row.get("appVersion", "Unknown"))
        if category == "Bug":
            severity, tech_details, _ = analyze_bug_with_gemini(g, text, platform, version)
            details = tech_details
            # map severity -> priority
            priority = {"Critical": "Critical", "High": "High", "Medium": "Medium", "Low": "Low"}.get(severity, "Medium")
            title_hint = f"Bug: {text[:72]}"
        elif category == "Feature Request":
            impact, feat_details, suggested_title = extract_feature_with_gemini(g, text)
            details = feat_details
            priority = {"High": "High", "Medium": "Medium", "Low": "Low"}.get(impact, "Low")
            title_hint = suggested_title or f"Feature: {text[:72]}"
        elif category == "Complaint":
            priority = "Medium"
            details = f"Complaint: {text[:240]}"
            title_hint = f"Complaint: {text[:72]}"
        elif category == "Praise":
            priority = "Low"
            details = "Positive feedback"
            title_hint = "Praise"
        elif category == "Spam":
            priority = "Low"
            details = "Likely spam"
            title_hint = "Spam"

        title, body = compose_ticket_with_gemini(g, title_hint, details)
        link_back = row.get("url", "")
        t = create_ticket(
            str(row.get("review_id", row.get("email_id", ""))),
            source_type,
            category,
            priority,
            title,
            body if body else details,
            float(conf),
            link_back,
        )
        return t

    processing_rows = []
    tickets = []

    # Process app store reviews
    for _, r in df_reviews.iterrows():
        t = row_to_ticket(r, "App Store Review")
        td = t.__dict__.copy()
        td = critic_with_gemini(g, td)
        tickets.append(td)
        processing_rows.append({
            "source_id": td["source_id"],
            "source_type": "App Store Review",
            "category": td["category"],
            "priority": td["priority"],
            "confidence": td["confidence"]
        })

    # Process support emails (optional)
    for _, r in df_support.iterrows():
        t = row_to_ticket(r, "Support Email")
        td = t.__dict__.copy()
        td = critic_with_gemini(g, td)
        tickets.append(td)
        processing_rows.append({
            "source_id": td["source_id"],
            "source_type": "Support Email",
            "category": td["category"],
            "priority": td["priority"],
            "confidence": td["confidence"]
        })

    tickets_df = pd.DataFrame(tickets)
    log_df = pd.DataFrame(processing_rows)

    expected_df = read_csv_file(expected_csv) if expected_csv and os.path.exists(expected_csv) else None
    metrics_df = compute_metrics(log_df, expected_df)

    save_df(tickets_df, files.get("generated_tickets", "generated_tickets.csv"))
    save_df(log_df, files.get("processing_log", "processing_log.csv"))
    save_df(metrics_df, files.get("metrics", "metrics.csv"))

    print("=== Done ===")
    print(f"Generated tickets: {len(tickets_df)}")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()
