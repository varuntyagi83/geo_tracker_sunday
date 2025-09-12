import time, re
from typing import Dict, Any, Optional, List

import google.generativeai as genai
from config import GOOGLE_API_KEY, GEMINI_DEFAULT_MODEL
from .base import LLMProvider

URL_RE = re.compile(r"https?://[^\s)\]]+")

def _dedupe_sources(sources: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for s in sources:
        url = (s.get("url") or "").strip()
        if url and url not in seen:
            out.append({"url": url, "title": s.get("title")})
            seen.add(url)
    return out

class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set")
        # Keep configuration minimal to avoid any side effects
        genai.configure(api_key=GOOGLE_API_KEY)

    def _extract_text(self, resp) -> str:
        try:
            return resp.text or ""
        except Exception:
            try:
                return resp.candidates[0].content.parts[0].text
            except Exception:
                return str(resp)

    def _extract_usage(self, resp):
        tokens_in = None
        tokens_out = None
        try:
            meta = getattr(resp, "usage_metadata", None)
            if meta:
                tokens_in = getattr(meta, "prompt_token_count", None)
                tokens_out = getattr(meta, "candidates_token_count", None)
        except Exception:
            pass
        return tokens_in, tokens_out

    def _extract_grounded_sources(self, resp, fallback_text: str) -> List[dict]:
        sources: List[dict] = []
        try:
            data = resp.to_dict() if hasattr(resp, "to_dict") else None
            if data:
                cands = data.get("candidates") or []
                for c in cands:
                    gm = c.get("groundingMetadata") or {}
                    for ch in gm.get("groundingChunks", []):
                        web = ch.get("web") or {}
                        url = web.get("uri") or web.get("url")
                        title = web.get("title")
                        if url:
                            sources.append({"url": url, "title": title})
                    for src in gm.get("citations", []):
                        url = src.get("uri") or src.get("url")
                        title = src.get("title")
                        if url:
                            sources.append({"url": url, "title": title})
        except Exception:
            pass
        if not sources and fallback_text:
            for m in URL_RE.finditer(fallback_text):
                sources.append({"url": m.group(0)})
        return _dedupe_sources(sources)

    def generate(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        model = model or GEMINI_DEFAULT_MODEL
        start = time.time()
        gmodel = genai.GenerativeModel(model)
        resp = gmodel.generate_content(prompt)
        latency_ms = int((time.time() - start) * 1000)

        text = self._extract_text(resp)
        tokens_in, tokens_out = self._extract_usage(resp)
        sources = self._extract_grounded_sources(resp, text)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": None,
            "sources": sources,
        }

    def generate_provider_web(self, prompt: str, model: Optional[str] = None,
                              dynamic_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Gemini built in grounded search for 1.5 models.
        """
        model = model or GEMINI_DEFAULT_MODEL
        start = time.time()
        gmodel = genai.GenerativeModel(model)
        tools = [{
            "google_search_retrieval": {
                "dynamic_retrieval_config": {
                    "mode": "MODE_DYNAMIC",
                    "dynamic_threshold": float(dynamic_threshold)
                }
            }
        }]
        resp = gmodel.generate_content(prompt, tools=tools)
        latency_ms = int((time.time() - start) * 1000)

        text = self._extract_text(resp)
        tokens_in, tokens_out = self._extract_usage(resp)
        sources = self._extract_grounded_sources(resp, text)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": None,
            "sources": sources,
        }
