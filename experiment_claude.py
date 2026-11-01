"""Baseline: classify each test text by prompting Claude, one request per text.

Reads datasets/<name>/data/{train,test}.parquet (train only for the label set),
asks Claude for one label per test text via structured output, and records
accuracy, macro-F1, per-request latency (mean, p90, p95, p99), tokens and cost.

Two backends:
    claude-code  (default) runs headless Claude Code (`claude -p`) on this machine's
                 claude.ai login, so it uses the subscription instead of an API key.
                 Cost is notional (Claude Code's estimate); requests count against the
                 plan's usage limits. latency_ms_* is Claude Code's model time; wall_ms_*
                 also includes the ~1.5 s CLI startup per request.
    api          calls the Messages API with the anthropic SDK; needs ANTHROPIC_API_KEY
                 (e.g. in the git-ignored .env-secrets). Cost is real money.

Writes to results/claude/:
    <dataset>_<model>_<backend>.csv              one summary row
    <dataset>_<model>_<backend>.json             run metadata (backend, model, prompt, sample, ...)
    <dataset>_<model>_<backend>_predictions.csv  one row per test text

    uv run experiment_claude.py --dataset sms_spam --limit 200
    uv run --env-file .env-secrets experiment_claude.py --backend api --model claude-opus-5-5 --max-cost 10
    uv run experiment_claude.py --dataset sms_spam --limit 200 --resume   # retry failed or unsent texts

The script prints an estimate, asks before sending requests, and stops sending
once (notional or real) spend in this run reaches --max-cost, or after
--max-failures replies in a row come back without a label (e.g. after hitting a
usage limit). Texts without a label are recorded as failures with NaN latency
and left out of the scores; --resume keeps the labelled rows of the previous
run's predictions and sends only the rest.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "claude"

TASKS = {  # dataset -> what the text is and what the labels mean
    "sms_spam": "an SMS message. Label it spam (unsolicited advertising, scams, prize or premium-rate offers) "
                "or ham (any legitimate personal or business message).",
    "emotion": "an English tweet. Label it with the main emotion its author expresses.",
}
PRICES = {  # model -> (input, output) USD per million tokens
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}  # api backend: accept server-side refusal fallbacks
EST_TOKENS = (250, 200)  # rough (input, output) tokens per API request; output includes thinking
CLAUDE_CODE_COST_FACTOR = 2  # claude -p costs roughly 2x a direct API call (extra structured-output turn)


def load(dataset: str, split: str) -> pd.DataFrame:
    path = ROOT / "datasets" / dataset / "data" / f"{split}.parquet"
    if not path.exists():
        sys.exit(f"{path} not found; run: uv run datasets/{dataset}/download.py")
    return pd.read_parquet(path)


def sample(test: pd.DataFrame, limit: int | None, seed: int = 0) -> pd.DataFrame:
    """The whole test set, or a stratified random subset of about `limit` texts."""
    if limit is None or limit >= len(test):
        return test.reset_index(drop=True)
    sub = test.groupby("label").sample(frac=limit / len(test), random_state=seed)
    return sub.sample(frac=1, random_state=seed).reset_index(drop=True)


def label_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": labels}},
        "required": ["label"],
        "additionalProperties": False,
    }


def failed(stop_reason: str, wall_ms: float) -> dict:
    """A request that produced no label. Its latency is NaN so it never enters the latency stats."""
    return {"pred": None, "latency_ms": np.nan, "wall_ms": wall_ms, "stop_reason": stop_reason,
            "served_by": None, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


RESULT_COLUMNS = list(failed("", 0.0))  # per-text fields every classify() returns


# --- Backend: Messages API ----------------------------------------------------


def api_classifier(model: str, effort: str, system: str, labels: list[str]) -> Callable[[str], dict]:
    import anthropic

    client = anthropic.Anthropic(max_retries=5)
    request = {
        "model": model,
        "max_tokens": 1024,
        "system": system,
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": label_schema(labels)}},
    }
    if model in FALLBACK_MODELS:
        request |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    price_in, price_out = PRICES[model]

    def classify(text: str) -> dict:
        start = time.perf_counter()
        try:
            response = client.beta.messages.create(**request, messages=[{"role": "user", "content": text}])
        except anthropic.APIStatusError as e:
            return failed(f"error {e.status_code}", (time.perf_counter() - start) * 1000)
        wall_ms = (time.perf_counter() - start) * 1000

        pred = None
        if response.stop_reason == "end_turn":
            body = next((b.text for b in response.content if b.type == "text"), "")
            try:
                pred = json.loads(body)["label"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        tokens_in, tokens_out = response.usage.input_tokens, response.usage.output_tokens
        return {
            "pred": pred, "latency_ms": wall_ms if pred is not None else np.nan, "wall_ms": wall_ms,
            "stop_reason": response.stop_reason,
            "served_by": response.model, "input_tokens": tokens_in, "output_tokens": tokens_out,
            "cost_usd": (tokens_in * price_in + tokens_out * price_out) / 1e6,
        }

    return classify


# --- Backend: headless Claude Code --------------------------------------------


def claude_code_classifier(model: str, effort: str, system: str, labels: list[str]) -> Callable[[str], dict]:
    cwd = tempfile.mkdtemp(prefix="llsvm-claude-")  # no project files or CLAUDE.md in reach
    cmd = [
        "claude", "-p",
        "--model", model,
        "--effort", effort,
        "--system-prompt", system,
        "--json-schema", json.dumps(label_schema(labels)),
        "--tools", "",
        "--setting-sources=",  # keep user/project settings and CLAUDE.md out of the context
        "--strict-mcp-config",  # no MCP servers
        "--no-session-persistence",
        "--output-format", "json",
    ]

    def classify(text: str) -> dict:
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, input=text, capture_output=True, text=True, cwd=cwd, timeout=300)
        except subprocess.TimeoutExpired:
            return failed("timeout", (time.perf_counter() - start) * 1000)
        wall_ms = (time.perf_counter() - start) * 1000
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return failed(f"exit {proc.returncode}", wall_ms)

        label = None if out.get("is_error") else (out.get("structured_output") or {}).get("label")
        if label is None:  # e.g. a usage-limit reply: subtype "success" but no label, no tokens, no model time
            return failed(f"no label ({out.get('subtype', 'unknown')})", wall_ms)

        usage = out.get("usage") or {}
        return {
            "pred": label,
            "latency_ms": out.get("duration_api_ms", wall_ms),
            "wall_ms": wall_ms,
            "stop_reason": out.get("subtype", "unknown"),
            "served_by": ",".join(out.get("modelUsage") or {}) or None,
            "input_tokens": sum(usage.get(k, 0) for k in
                                ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
            "output_tokens": usage.get("output_tokens", 0),
            "cost_usd": out.get("total_cost_usd", 0.0),
        }

    return classify


BACKENDS = {"claude-code": claude_code_classifier, "api": api_classifier}


# --- Main ---------------------------------------------------------------------


def percentiles(prefix: str, ms: np.ndarray) -> dict:
    stats = ("mean", "p90", "p95", "p99")
    if not len(ms):
        return {f"{prefix}_{s}": np.nan for s in stats}
    return {f"{prefix}_mean": ms.mean(), **{f"{prefix}_p{q}": np.percentile(ms, q) for q in (90, 95, 99)}}


def previous_run(stem: str, effort: str, system: str) -> tuple[dict[str, dict], float]:
    """For --resume: the labelled rows of the last run, by text, and that run's predict_seconds."""
    paths = [OUT / f"{stem}{suffix}" for suffix in ("_predictions.csv", ".json", ".csv")]
    if missing := [p.name for p in paths if not p.exists()]:
        sys.exit(f"--resume: no previous run to resume (missing {', '.join(missing)} in results/claude/)")
    meta = json.loads(paths[1].read_text())
    if (meta["effort"], meta["system_prompt"]) != (effort, system):
        sys.exit("--resume: the previous run used a different effort or system prompt; rerun without --resume")
    # Only an empty cell is missing: a tweet that reads "NA" or "null" must stay text.
    old = pd.read_csv(paths[0], keep_default_na=False, na_values=[""], dtype={"text": str, "label": str, "pred": str})
    labelled = old[old["pred"].notna()]
    rows = {text: dict(zip(RESULT_COLUMNS, values))
            for text, *values in labelled[["text", *RESULT_COLUMNS]].itertuples(index=False)}
    return rows, float(pd.read_csv(paths[2], index_col="classifier")["predict_seconds"].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=TASKS, default="sms_spam")
    parser.add_argument("--backend", choices=BACKENDS, default="claude-code")
    parser.add_argument("--model", choices=PRICES, default="claude-sonnet-5-5")
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--limit", type=int, default=None, help="stratified sample of about N test texts (default: all)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="requests in flight at once (default: 4 for claude-code, 8 for api)")
    parser.add_argument("--max-cost", type=float, default=3.0,
                        help="stop sending requests once spend in this run reaches this many USD (default: 3)")
    parser.add_argument("--max-failures", type=int, default=10,
                        help="stop sending requests after this many replies in a row without a label (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="keep the labelled rows of the previous run with the same dataset, model and backend, "
                             "and send only the failed or unsent texts")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()
    concurrency = args.concurrency or (4 if args.backend == "claude-code" else 8)

    labels = sorted(load(args.dataset, "train")["label"].unique())
    full_test = load(args.dataset, "test")
    test = sample(full_test, args.limit)
    system = (f"You are a text classifier. Each user message is {TASKS[args.dataset]} "
              f"Answer with exactly one label from: {', '.join(labels)}.")
    stem = f"{args.dataset}_{args.model}_{args.backend}"

    previous, previous_seconds = previous_run(stem, args.effort, system) if args.resume else ({}, 0.0)
    rows: list[dict | None] = [previous.get(text) for text in test["text"]]
    todo_items = [(i, text) for i, text in enumerate(test["text"]) if rows[i] is None]
    reused = len(test) - len(todo_items)

    price_in, price_out = PRICES[args.model]
    estimate = len(todo_items) * (EST_TOKENS[0] * price_in + EST_TOKENS[1] * price_out) / 1e6
    if args.backend == "claude-code":
        estimate *= CLAUDE_CODE_COST_FACTOR
    cost_kind = "notional, counts against your Claude plan's usage limits" if args.backend == "claude-code" else "billed"
    print(f"{args.dataset}: {len(test):,} test texts, labels {labels}")
    if args.resume:
        print(f"resuming: {reused:,} texts already labelled, {len(todo_items):,} to send")
    print(f"backend {args.backend}, model {args.model}, effort {args.effort}, concurrency {concurrency}")
    print(f"rough cost estimate ${estimate:.2f} ({cost_kind}), hard cap ${args.max_cost:.2f}")
    if todo_items and not args.yes and input("Send these requests? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    classify = BACKENDS[args.backend](args.model, args.effort, system, labels)
    todo = iter(todo_items)
    spent, capped, failure_streak, halted = 0.0, False, 0, False
    start = time.perf_counter()
    with (ThreadPoolExecutor(concurrency) as pool,
          tqdm(total=len(test), initial=reused, desc=args.model, unit="req") as bar):
        pending = {}

        def submit_next() -> None:
            if (item := next(todo, None)) is not None:
                pending[pool.submit(classify, item[1])] = item[0]

        for _ in range(concurrency):
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                i = pending.pop(future)
                rows[i] = future.result()
                spent += rows[i]["cost_usd"]
                failure_streak = failure_streak + 1 if rows[i]["pred"] is None else 0
                bar.update()
                bar.set_postfix_str(f"${spent:.2f}")
                if spent >= args.max_cost:
                    capped = True
                elif failure_streak >= args.max_failures and not halted:
                    halted = True
                    tqdm.write(f"{failure_streak} replies in a row without a label "
                               f"(last: {rows[i]['stop_reason']}); not sending more")
                elif not capped and not halted:
                    submit_next()
    wall_seconds = previous_seconds + time.perf_counter() - start

    finished = [i for i, r in enumerate(rows) if r is not None]
    if capped:
        print(f"stopped at the ${args.max_cost:.2f} cap after {len(finished):,} of {len(test):,} texts")
    test = test.iloc[finished].reset_index(drop=True)
    preds = pd.concat([test, pd.DataFrame([rows[i] for i in finished], columns=RESULT_COLUMNS)], axis=1)

    # A text without a label is an infrastructure failure (usage limit, timeout, error), not a wrong answer:
    # it is counted in n_failed and left out of the scores and latency stats, and --resume retries it.
    answered = preds[preds["pred"].notna()]
    n_failed = len(preds) - len(answered)
    y_true, y_pred = answered["label"], answered["pred"]
    summary = pd.DataFrame([{
        "classifier": args.model,
        "family": "LLM",
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro") if len(answered) else np.nan,
        "accuracy": accuracy_score(y_true, y_pred) if len(answered) else np.nan,
        "train_seconds": 0.0,
        "predict_seconds": wall_seconds,
        **percentiles("latency_ms", answered["latency_ms"].to_numpy(dtype=float)),
        **percentiles("wall_ms", answered["wall_ms"].to_numpy(dtype=float)),
        "n_failed": n_failed,
        "input_tokens": int(preds["input_tokens"].sum()),
        "output_tokens": int(preds["output_tokens"].sum()),
        "cost_usd": preds["cost_usd"].sum(),
    }]).set_index("classifier")
    print(summary.round(4).T.to_string())
    if n_failed or len(finished) < len(rows):
        print(f"{n_failed:,} texts failed and {len(rows) - len(finished):,} were not sent; "
              f"rerun with the same arguments plus --resume to fill them in")

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / f"{stem}.csv", lineterminator="\n")
    preds.to_csv(OUT / f"{stem}_predictions.csv", index=False, lineterminator="\n")
    (OUT / f"{stem}.json").write_text(json.dumps({
        "dataset": args.dataset,
        "backend": args.backend,
        "model": args.model,
        "effort": args.effort,
        "n_test": len(test),
        "n_scored": len(answered),
        "test_subsample": len(test) < len(full_test),
        "resumed_rows": reused,
        "stopped_at_cost_cap": capped,
        "stopped_after_failures": halted,
        "max_cost_usd": args.max_cost,
        "cost_is_notional": args.backend == "claude-code",
        "latency_ms_is": "model time reported by Claude Code" if args.backend == "claude-code" else "request wall time",
        "concurrency": concurrency,
        "system_prompt": system,
        "served_by": preds["served_by"].value_counts().to_dict(),
        "stop_reasons": preds["stop_reason"].value_counts().to_dict(),
    }, indent=2) + "\n")
    print(f"wrote results/claude/{stem}.csv, .json and _predictions.csv")


if __name__ == "__main__":
    main()
