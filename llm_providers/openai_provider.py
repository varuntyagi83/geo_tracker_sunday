import time, re
from typing import Dict, Any, Optional, List

from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_DEFAULT_MODEL
from .base import LLMProvider

URL_RE = re.compile(r"https?://[^\s)\]]+")

def _dedupe_sources_dict(sources: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for s in sources:
        url = (s.get("url") or "").strip()
        if url and url not in seen:
            out.append({"url": url, "title": s.get("title")})
            seen.add(url)
    return out

class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    # ---------- helpers ----------
    def _extract_usage_chat(self, resp):
        try:
            usage = resp.usage
            return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
        except Exception:
            return None, None

    def _extract_usage_responses(self, resp):
        try:
            usage = getattr(resp, "usage", None)
            if usage:
                # responses API often exposes input_tokens/output_tokens
                ti = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
                to = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
                return ti, to
        except Exception:
            pass
        return None, None

    def _extract_text_responses(self, resp) -> str:
        # Fast path (SDK helper)
        text = getattr(resp, "output_text", None)
        if text:
            return text
        # Robust path
        try:
            data = resp.model_dump()
            # Find first message text
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if "text" in part and part["text"]:
                            return part["text"]
            # Fallback: whole dump
            return str(data)
        except Exception:
            return str(resp)

    def _extract_citations_responses(self, resp, fallback_text: str) -> List[dict]:
        sources: List[dict] = []
        try:
            data = resp.model_dump()
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        for ann in part.get("annotations") or []:
                            t = ann.get("type")
                            if t in ("url_citation", "web_search.url_citation", "web_search_citation"):
                                url = ann.get("url") or ann.get("href")
                                title = ann.get("title")
                                if url:
                                    sources.append({"url": url, "title": title})
        except Exception:
            pass

        # Fallback: parse any URLs mentioned in the answer text
        if not sources and fallback_text:
            for m in URL_RE.finditer(fallback_text):
                sources.append({"url": m.group(0)})

        return _dedupe_sources_dict(sources)

    # ---------- public API ----------
    def generate(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        INTERNAL mode: standard chat completion (no live web).
        """
        model = model or OPENAI_DEFAULT_MODEL
        start = time.time()
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        latency_ms = int((time.time() - start) * 1000)
        text = (resp.choices[0].message.content or "").strip()
        tokens_in, tokens_out = self._extract_usage_chat(resp)

        # Extract any URLs from the text as "sources" for parity
        sources = [{"url": u} for u in URL_RE.findall(text)]
        sources = _dedupe_sources_dict(sources)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": None,
            "sources": sources,
        }

    def generate_provider_web(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        PROVIDER_WEB mode: use OpenAI Responses API with built-in web search.
        Prefers 'web_search', falls back to 'web_search_preview' if needed.
        """
        model = model or OPENAI_DEFAULT_MODEL
        start = time.time()

        # Try the current tool name first
        tools_primary = [{"type": "web_search"}]
        tools_fallback = [{"type": "web_search_preview"}]

        try:
            resp = self.client.responses.create(model=model, input=prompt, tools=tools_primary)
        except Exception:
            # Fallback to preview tool if primary not enabled on your account/model
            try:
                resp = self.client.responses.create(model=model, input=prompt, tools=tools_fallback)
            except Exception:
                # Last resort: degrade to internal generate()
                return self.generate(prompt, model=model)

        latency_ms = int((time.time() - start) * 1000)
        text = self._extract_text_responses(resp)
        tokens_in, tokens_out = self._extract_usage_responses(resp)
        sources = self._extract_citations_responses(resp, text)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": None,
            "sources": sources,
        }
