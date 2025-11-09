import json, random, subprocess, sys, os, time, multiprocessing as mp

BASE = {
 "W_SPACE":1.30,"W_SPACE_REPLY":0.35,"W_BRANCH":0.22,"W_TUNNEL":-0.20,"W_STRAIGHT":0.05,
 "W_CHOKE":-0.45,"W_HEADON_WIN":0.30,"W_VORONOI":0.40,"W_VORONOI_RPLY":0.20,
 "W_ARTICULATE":-0.50,"W_THREAT2":-0.25,
 "EPS_NEAR_TIE":3.0,"BOOST_DELTA_T":8
}

def write_weights(dct, path="weights.json"):
    with open(path,"w") as f: json.dump(dct,f)

def run_eval():
    # Run a short batch; parse "Final: {'agent1': X, 'agent2': Y, 'draw': Z}"
    p = subprocess.run([sys.executable,"run_batch.py","60"], capture_output=True, text=True)
    out = p.stdout + "\n" + p.stderr
    # crude parse
    import re
    m = re.search(r"Final:\s*\{.*\}", out)
    if not m: return 0.5
    block = m.group(0)
    nums = list(map(int, re.findall(r"\d+", block)))
    if len(nums) < 3: return 0.5
    a1,a2,dr = nums[-3],nums[-2],nums[-1]
    total = a1+a2+dr if (a1+a2+dr)>0 else 1
    return a1/total

def worker(seed):
    rnd = random.Random(seed)
    cand = dict(BASE)
    # small perturbations
    for k in ["W_SPACE","W_VORONOI","W_BRANCH","W_TUNNEL","W_CHOKE","W_VORONOI_RPLY","W_SPACE_REPLY"]:
        cand[k] = round(cand[k] * (1.0 + rnd.uniform(-0.25,0.25)), 3)
    # write and run
    write_weights(cand, "weights.json")
    score = run_eval()
    return (score, cand)

def main():
    N = int(os.getenv("TRIALS","24"))
    with mp.Pool(processes=min(24, mp.cpu_count())) as pool:
        res = pool.map(worker, range(N))
    best = max(res, key=lambda x:x[0])
    print("BEST:", best[0], best[1])
    write_weights(best[1], "weights_best.json")

if __name__ == "__main__":
    main()
