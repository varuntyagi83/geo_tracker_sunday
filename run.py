import argparse, json, time, sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from gsheets import read_prompts_dataframe
from llm_providers.openai_provider import OpenAIProvider
from llm_providers.gemini_provider import GeminiProvider
from metrics.presence import compute_presence_rate
from metrics.sentiment import compute_sentiment
from metrics.trust import compute_trustworthiness
from db import init_db, insert_run, insert_response, insert_metrics
from config import OPENAI_DEFAULT_MODEL, GEMINI_DEFAULT_MODEL

PROVIDERS = {"openai": OpenAIProvider, "gemini": GeminiProvider}

def _call_with_timeout(fn, timeout_s: int, retries: int, label: str):
    last_err = None
    for attempt in range(1, retries + 2):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn)
                return fut.result(timeout=timeout_s)
        except TimeoutError:
            print(f"[timeout] {label} attempt {attempt} exceeded {timeout_s}s", file=sys.stderr)
            last_err = TimeoutError(f"{label} timed out")
        except Exception as e:
            print(f"[error] {label} attempt {attempt} failed: {e}", file=sys.stderr)
            last_err = e
        time.sleep(1.0 * attempt)
    return {"text": "", "latency_ms": None, "tokens_in": None, "tokens_out": None, "cost_usd": None, "sources": [], "error": str(last_err) if last_err else None}

def _preview(text: str, max_len: int) -> str:
    if max_len <= 0 or not text:
        return ""
    s = " ".join(str(text).split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def execute_all(provider_name: str, model: str, mode: str,
                limit: int | None = None, start: int = 0,
                market: str | None = None, lang: str | None = None,
                raw: bool = False,
                request_timeout: int = 60,
                max_retries: int = 1,
                sleep_ms: int = 0,
                log_question_len: int = 160):
    """
    Modes for both providers:
      internal      -> model knowledge only
      provider_web  -> provider-native web tools if available
    """
    init_db()
    df = read_prompts_dataframe()

    if start or limit:
        end = start + limit if limit else None
        df = df.iloc[start:end].reset_index(drop=True)
        print(f"[runner] Using slice start={start}, limit={limit or 'ALL'}, rows={len(df)}")

    ProviderCls = PROVIDERS[provider_name]
    provider = ProviderCls()

    if not model:
        model = OPENAI_DEFAULT_MODEL if provider_name == "openai" else GEMINI_DEFAULT_MODEL

    total = len(df)
    for idx, row in df.iterrows():
        prompt_id = row.get("prompt_id") or ""
        category  = row.get("category") or ""
        question  = row.get("question") or ""
        metric    = row.get("metric") or ""

        qprev = _preview(question, log_question_len)
        qpart = f" • q={qprev!r}" if qprev else ""
        print(f"[{idx+1}/{total}] run start • prompt_id={prompt_id}{qpart} • mode={mode} • raw={raw}")

        # Build final payload
        if raw:
            prompt_text = question
        else:
            header = ""
            if mode == "provider_web" and (market or lang):
                header = f"(Market: {market or '-'}; Language: {lang or '-'})\n\n"
            prompt_text = header + question

        # NOTE: store canonical mode only; raw flag goes into extra JSON
        run_id = insert_run(
            provider    = provider_name,
            model       = model,
            prompt_id   = prompt_id,
            category    = category,
            mode        = mode,          # <— canonical only
            question    = question,
            prompt_text = prompt_text,
            market      = market,
            lang        = lang,
            extra       = {"raw": bool(raw)}
        )
        print(f"[{idx+1}/{total}] run_id={run_id} inserted")

        # Provider call
        label = f"{provider_name}:{mode}"
        if mode == "provider_web" and hasattr(provider, "generate_provider_web"):
            result = _call_with_timeout(lambda: provider.generate_provider_web(prompt_text, model=model), request_timeout, max_retries, label)
        else:
            result = _call_with_timeout(lambda: provider.generate(prompt_text, model=model), request_timeout, max_retries, label)

        response_text    = result.get("text", "") or ""
        latency_ms       = result.get("latency_ms")
        tokens_in        = result.get("tokens_in")
        tokens_out       = result.get("tokens_out")
        cost_usd         = result.get("cost_usd")
        provider_sources = result.get("sources") or []

        insert_response(
            run_id=run_id,
            response_text=response_text,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            provider_sources=provider_sources
        )
        print(f"[{idx+1}/{total}] response saved • sources={len(provider_sources)} • chars={len(response_text)}")

        # Metrics only when presence is expected
        if str(metric).strip():
            presence = compute_presence_rate(response_text, metric)
            sentiment = compute_sentiment(response_text) if presence is not None else None
            trust_authority, trust_sunday = compute_trustworthiness(response_text, provider_sources)
            details = {"metric_field": metric, "market": market, "lang": lang, "raw": bool(raw)}
            try:
                insert_metrics(run_id, presence, sentiment, trust_authority, trust_sunday, details)
            except TypeError:
                insert_metrics(run_id, presence, sentiment, trust_authority, details)
            print(f"[{idx+1}/{total}] metrics saved • presence={presence} • sentiment={sentiment} • trustA={trust_authority} • trustS={trust_sunday}")
        else:
            print(f"[{idx+1}/{total}] metrics skipped • presence not expected")

        if sleep_ms:
            time.sleep(sleep_ms / 1000.0)

    print("[runner] all done")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "gemini"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["internal", "provider_web"], default="internal")
    ap.add_argument("--interval-minutes", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--market", type=str, default=None)
    ap.add_argument("--lang", type=str, default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--request-timeout", type=int, default=60)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--sleep-ms", type=int, default=0)
    ap.add_argument("--log-question-len", type=int, default=160)
    args = ap.parse_args()

    model = args.model or (OPENAI_DEFAULT_MODEL if args.provider == "openai" else GEMINI_DEFAULT_MODEL)

    if args.interval_minutes > 0:
        print(f"Looping every {args.interval_minutes} minutes. Ctrl+C to stop.")
        while True:
            execute_all(args.provider, model, args.mode,
                        limit=args.limit, start=args.start,
                        market=args.market, lang=args.lang,
                        raw=args.raw,
                        request_timeout=args.request_timeout,
                        max_retries=args.max_retries,
                        sleep_ms=args.sleep_ms,
                        log_question_len=args.log_question_len)
            time.sleep(args.interval_minutes * 60)
    else:
        execute_all(args.provider, model, args.mode,
                    limit=args.limit, start=args.start,
                    market=args.market, lang=args.lang,
                    raw=args.raw,
                    request_timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    sleep_ms=args.sleep_ms,
                    log_question_len=args.log_question_len)

if __name__ == "__main__":
    main()
