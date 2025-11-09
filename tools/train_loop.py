#!/usr/bin/env python3
"""
Automated training loop (with live output + heartbeat)
- runs N generations
- tunes BOOST_DELTA_T
- evaluates via self-play
- promotes best
- streams progress so it's clear the run is alive
"""

import os, sys, subprocess, time, random, json, threading, queue, itertools

AGENT_A_PORT = 5008
AGENT_B_PORT = 5009
A_URL = f"http://127.0.0.1:{AGENT_A_PORT}"
B_URL = f"http://127.0.0.1:{AGENT_B_PORT}"

# --- TRAINING CONFIG ---
GAMES_PER_GEN = 200       # batch size per generation
WORKERS = 8               # parallel workers
GENERATIONS = 10          # total gens
SEARCH_RANGE = (4, 14)    # boost tuning search range
AGENT_A_NAME = "AgentA"
AGENT_B_NAME = "AgentB"
LOG_DIR = "tools/out"
BOOST_FILE = f"{LOG_DIR}/boost_history.json"

HEARTBEAT_SECS = 1.0      # spinner period while process is quiet
QUIET_WARN_SECS = 30.0    # if no output for this long, print a warning line

PY = sys.executable
os.makedirs(LOG_DIR, exist_ok=True)

# ---------- utilities ----------

def popen_stream(cmd, env=None, cwd=None, title="process"):
    """
    Run a process, stream stdout/stderr live, and show a heartbeat spinner when quiet.
    Returns (returncode, captured_stdout, captured_stderr).
    """
    print(f"\n[spawn] {title}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    q_out, q_err = queue.Queue(), queue.Queue()
    captured_out, captured_err = [], []
    last_line_time = time.time()

    def pump(stream, q, tag, cap):
        for line in iter(stream.readline, ''):
            cap.append(line)
            q.put((tag, line))
        stream.close()

    t_out = threading.Thread(target=pump, args=(proc.stdout, q_out, "OUT", captured_out), daemon=True)
    t_err = threading.Thread(target=pump, args=(proc.stderr, q_err, "ERR", captured_err), daemon=True)
    t_out.start(); t_err.start()

    spinner = itertools.cycle("|/-\\")
    last_warn = 0.0

    # Merge queues by polling both
    while True:
        did_print = False
        # Drain stdout
        while not q_out.empty():
            tag, line = q_out.get_nowait()
            print(f"[{title}][stdout] {line.rstrip()}")
            last_line_time = time.time()
            did_print = True
        # Drain stderr
        while not q_err.empty():
            tag, line = q_err.get_nowait()
            print(f"[{title}][stderr] {line.rstrip()}")
            last_line_time = time.time()
            did_print = True

        if proc.poll() is not None:
            # flush any remaining
            while not q_out.empty():
                tag, line = q_out.get_nowait()
                print(f"[{title}][stdout] {line.rstrip()}")
                captured_out.append(line)
            while not q_err.empty():
                tag, line = q_err.get_nowait()
                print(f"[{title}][stderr] {line.rstrip()}")
                captured_err.append(line)
            break

        # Heartbeat if quiet
        if not did_print:
            elapsed_quiet = time.time() - last_line_time
            sys.stdout.write(f"\r[{title}] {next(spinner)} alive… quiet {int(elapsed_quiet)}s")
            sys.stdout.flush()
            if elapsed_quiet - last_warn >= QUIET_WARN_SECS:
                print(f"\n[{title}] still running (no output for {int(elapsed_quiet)}s)…")
                last_warn = elapsed_quiet
            time.sleep(HEARTBEAT_SECS)

    rc = proc.returncode
    print(f"\n[done] {title} exited with code {rc}")
    return rc, "".join(captured_out), "".join(captured_err)


def run_selfplay(boost_val):
    env = os.environ.copy()
    env["BOOST_DELTA_T"] = str(boost_val)
    # You may also pass other knobs via env here (e.g., SAFETY_MARGIN, etc.)
    cmd = [
        PY, "tools/selfplay.py",
        "--p1", A_URL, "--p2", B_URL,
        "-n", str(GAMES_PER_GEN),
        "-w", str(WORKERS),
        "--out", LOG_DIR
    ]
    return popen_stream(cmd, env=env, title="selfplay")


def run_elo():
    cmd = [
        PY, "tools/elo.py",
        "--p1_name", AGENT_A_NAME,
        "--p2_name", AGENT_B_NAME,
        "--k", "24"
    ]
    return popen_stream(cmd, title="elo")


def evaluate(boost_val):
    print(f"\n[Eval] BOOST_DELTA_T = {boost_val}")
    t0 = time.time()
    rc, so, se = run_selfplay(boost_val)
    if rc != 0:
        print("[warn] selfplay failed; showing stderr tail:")
        print(se.splitlines()[-30:])
        # Return a conservative elo so it won't be chosen as best
        return 0

    rc, so, se = run_elo()
    if rc != 0:
        print("[warn] elo failed; showing stderr tail:")
        print(se.splitlines()[-30:])
        return 0

    try:
        j = json.loads(so)
        elo = j["ratings"].get(AGENT_A_NAME, 1000)
    except Exception as e:
        print("[warn] failed to parse ELO JSON; defaulting to 1000")
        print("stdout:", so[-500:])
        print("stderr:", se[-500:])
        elo = 1000

    dt = time.time() - t0
    print(f"[Eval] BOOST={boost_val} → Elo={elo} (took {dt:.1f}s)")
    return elo


def choose_random_boost():
    return random.randint(SEARCH_RANGE[0], SEARCH_RANGE[1])


def main():
    print("===== AUTO TRAIN LOOP START (with live progress) =====")
    print(f"Config: GAMES_PER_GEN={GAMES_PER_GEN} WORKERS={WORKERS} GENERATIONS={GENERATIONS}")
    print(f"Agents: {AGENT_A_NAME}@{A_URL} vs {AGENT_B_NAME}@{B_URL}")
    print(f"Search: BOOST in {SEARCH_RANGE}")

    history = []
    best = {"gen": 0, "boost": None, "elo": float("-inf")}

    for gen in range(1, GENERATIONS + 1):
        print(f"\n================ Generation {gen}/{GENERATIONS} ================")
        boost = choose_random_boost()
        elo = evaluate(boost)

        record = {"gen": gen, "boost": boost, "elo": elo, "ts": time.time()}
        history.append(record)

        # Save after each gen for safety
        with open(BOOST_FILE, "w") as f:
            json.dump(history, f, indent=2)

        if elo > best["elo"]:
            best = record
            print(f"[Gen {gen}] 🔼 New best → BOOST={best['boost']} Elo={best['elo']}")
        else:
            print(f"[Gen {gen}] Result → BOOST={boost} Elo={elo} | Best so far BOOST={best['boost']} Elo={best['elo']}")

        # Apply best for next runs (via env) so downstream tools pick it up
        os.environ["BOOST_DELTA_T"] = str(best["boost"]) if best["boost"] is not None else str(boost)

    print("\n===== TRAIN COMPLETE =====")
    print(f"Best overall: BOOST={best['boost']} Elo={best['elo']}")
    print(f"History saved → {BOOST_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[exit] Interrupted by user.")
        sys.exit(130)
