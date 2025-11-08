# run_batch.py
import re, subprocess, sys, os

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 50

# Match formats like:
#   "Winner: Agent 1 (AgentX)"
#   "Winner: Agent 2 (SampleAgent)"
#   "Winner: draw" / "It's a draw"
winner_patterns = [
    re.compile(r"Winner:\s*Agent\s*(\d)", re.IGNORECASE),
    re.compile(r"Winner:\s*draw", re.IGNORECASE),
    re.compile(r"\bIt'?s a draw\b", re.IGNORECASE),
]

wins = {"agent1": 0, "agent2": 0, "draw": 0}
fail = 0

for i in range(GAMES):
    p = subprocess.run([sys.executable, "judge_engine.py"], capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    result = None

    for pat in winner_patterns:
        m = pat.search(out)
        if m:
            if pat.pattern.lower().startswith("winner:"):
                if m.groups():
                    num = m.group(1)
                    result = f"agent{num}"
                else:
                    result = "draw"
            else:
                result = "draw"
            break

    if result in wins:
        wins[result] += 1
    else:
        fail += 1
        # Uncomment to debug unparsed outputs:
        # print(f"[{i+1}] could not parse winner. Exit={p.returncode}")
        # print(out)

    if (i + 1) % 10 == 0:
        print(f"After {i+1} games: {wins} (unparsed: {fail})")

print("Final:", wins, f"(unparsed: {fail})")
