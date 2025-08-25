# gemini_client.py
import os
import json
from typing import Optional, Dict, Any
import google.generativeai as genai

class GeminiClient:
    def __init__(self, model: str = "gemini-1.5-flash", temperature: float = 0.2, top_p: float = 0.95, top_k: int = 40):
        api_key = os.getenv("GEMINI_API_KEY")
        self.enabled = api_key is not None and len(api_key.strip()) > 0
        self.model_name = model
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        if self.enabled:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
        else:
            self.model = None

    def ask_json(self, system_prompt: str, user_content: str) -> Optional[Dict[str, Any]]:
        """
        Sends a structured instruction to Gemini and expects a JSON-ish reply.
        Returns dict or None on failure.
        """
        if not self.enabled:
            return None
        prompt = f"{system_prompt}\n\nUser Input:\n{user_content}\n\nReturn a valid JSON only."
        resp = self.model.generate_content([prompt], generation_config=self.generation_config)
        text = (resp.text or "").strip()
        # Try to extract JSON (Gemini typically returns valid JSON when asked)
        try:
            return json.loads(text)
        except Exception:
            # Best-effort JSON extraction (very light)
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(text[start:end+1])
            except Exception:
                return None
        return None
