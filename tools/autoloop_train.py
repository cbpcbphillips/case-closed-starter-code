# tools/autoloop_train.py
"""
Auto-loop trainer: repeatedly runs fast_train in small chunks and promotes the best weights.
- Each cycle runs a small candidate-vs-baseline sweep (e.g., 10 games).
- Copies weights_best.json -> weights.json if produced.
- Repeats until total target games reached.
- Keeps a compact JSONL log in tools/autoloop_log.jsonl.

Run:
  python tools/autoloop_train.py --total-games 300 --chunk 10 --trials 8 --workers 8
"""

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
FAST_TRAIN = TOOLS / "fast_train.py"
LOG_PATH = TOOLS / "autoloop_log.jsonl"
WEIGHTS = ROOT / "weights.json"
WEIGHTS_BEST = ROOT / "weights_best.json"

def run_fast_train(chunk_games: int, trials: int, workers: int, baseline: str | None, extra_env=None) -> dict:
    """
    Invoke tools/fast_train.py as a subprocess and capture summarized info.
    Returns a dict with keys: exit_code, stdout, stderr, timestamp.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    args = [
        sys.executable, str(FAST_TRAIN),
        "--games", str(chunk_games),
        "--trials", str(trials),
        "--workers", str(workers),
    ]
    if baseline:
        args += ["--baseline", baseline]
    p = subprocess.run(args, cwd=str(ROOT), env=env, capture_output=True, text=True)
    return {
        "exit_code": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "timestamp": time.time(),
        "args": args,
    }

def promote_if_present() -> bool:
    if WEIGHTS_BEST.exists():
        copyfile(WEIGHTS_BEST, WEIGHTS)
        return True
    return False

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-games", type=int, default=200, help="Total games budget across the loop.")
    ap.add_argument("--chunk", type=int, default=10, help="Games per fast_train call.")
    ap.add_argument("--trials", type=int, default=8, help="Trials per chunk (parallel candidates).")
    ap.add_argument("--workers", type=int, default=8, help="Worker processes for fast_train.")
    ap.add_argument("--baseline", type=str, default="weights.json", help="Baseline file path for fast_train.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between chunks.")
    ap.add_argument("--log", type=str, default=str(LOG_PATH), help="Path to JSONL log.")
    ap.add_argument("--no-train-mode", action="store_true", help="Disable HEURISTICS_TRAIN exploration.")
    return ap.parse_args()

def append_log(log_file: Path, record: dict):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    args = parse_args()
    total = max(1, args.total_games)
    chunk = max(1, args.chunk)
    cycles = (total + chunk - 1) // chunk

    print(f"[AUTOLOOP] total_games={total}  chunk={chunk}  cycles={cycles}  trials={args.trials}  workers={args.workers}")
    print(f"[AUTOLOOP] baseline={args.baseline}  log={args.log}")

    log_file = Path(args.log)

    # Train-mode randomness for exploration
    extra_env = {}
    if not args.no_train_mode:
        extra_env["HEURISTICS_TRAIN"] = "1"
    else:
        extra_env.pop("HEURISTICS_TRAIN", None)

    games_done = 0
    for i in range(1, cycles + 1):
        # Clamp last chunk to hit exact total
        this_chunk = min(chunk, total - games_done)
        print(f"\n[CYCLE {i}/{cycles}] games={this_chunk}  (so far {games_done}/{total})")
        res = run_fast_train(
            chunk_games=this_chunk,
            trials=args.trials,
            workers=args.workers,
            baseline=args.baseline,
            extra_env=extra_env,
        )

        # Print short summary to console
        ec = res["exit_code"]
        print(f"[FAST_TRAIN exit={ec}]")
        if res["stdout"]:
            # print last ~10 lines to keep console tidy
            out_lines = res["stdout"].strip().splitlines()
            tail = "\n".join(out_lines[-10:])
            print(tail)
        if res["stderr"]:
            err_lines = res["stderr"].strip().splitlines()
            tail = "\n".join(err_lines[-6:])
            if tail:
                print("[stderr tail]")
                print(tail)

        # Promote best weights if produced
        promoted = promote_if_present()
        if promoted:
            print(f"[PROMOTE] {WEIGHTS_BEST.name} -> {WEIGHTS.name}")
        else:
            print("[PROMOTE] No weights_best.json found; skipping promotion.")

        # Log cycle
        record = {
            "cycle": i,
            "chunk_games": this_chunk,
            "timestamp": res["timestamp"],
            "exit_code": ec,
            "promoted": promoted,
        }
        # crude extraction of BEST and RECHECK lines for quick metrics
        if res["stdout"]:
            lines = res["stdout"].splitlines()
            best_line = next((ln for ln in lines if ln.startswith("[BEST] WR(candidate)=")), None)
            recheck_line = next((ln for ln in lines if ln.startswith("[RECHECK] WR(candidate)=")), None)
            record["best_line"] = best_line
            record["recheck_line"] = recheck_line
        append_log(log_file, record)

        games_done += this_chunk
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\n[DONE] Completed {games_done}/{total} games. Logs at: {log_file}")
    # Final deterministic recheck
    print("[FINAL] Deterministic baseline-only check...")
    env_final = os.environ.copy()
    env_final.pop("HEURISTICS_TRAIN", None)
    final = subprocess.run(
        [sys.executable, str(FAST_TRAIN), "--games", "200", "--trials", "1", "--workers", "1", "--baseline-only"],
        cwd=str(ROOT), env=env_final, capture_output=True, text=True
    )
    print(final.stdout or "")
    if final.stderr:
        print("[final stderr]", final.stderr)

if __name__ == "__main__":
    if not FAST_TRAIN.exists():
        raise FileNotFoundError(f"Missing {FAST_TRAIN}")
    main()
