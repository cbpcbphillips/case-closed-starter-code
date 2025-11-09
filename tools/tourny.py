import os, subprocess, sys, time, random

def run_one(seed: int) -> int:
    env = os.environ.copy()
    env["GAME_SEED"] = str(seed)
    p = subprocess.run([sys.executable, "judge_engine.py"], env=env)
    return p.returncode

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"Running {n} matches with varying seeds...")
    rc_bad = 0
    for i in range(1, n+1):
        seed = random.randint(1, 10_000_000)
        print(f"=== Match {i}/{n} (GAME_SEED={seed}) ===")
        rc = run_one(seed)
        if rc != 0: rc_bad += 1
        time.sleep(0.1)
    print(f"Done. Non-zero exits: {rc_bad}/{n}")

if __name__ == "__main__":
    main()
