"""Focused concurrency probe: N runs at a fixed client concurrency, median.

Kept in-repo so the numbers in docs/notes/m1-v3-bridge.md can be re-derived after
/tmp is cleaned.  Usage: python benchmarks/v3_conc.py <port> <out.json> <runs> <concurrency>
"""

import json, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
PORT=int(sys.argv[1]); OUT=sys.argv[2]; RUNS=int(sys.argv[3]); C=int(sys.argv[4])
BASE=f"http://127.0.0.1:{PORT}"
SHORT="用三句话介绍分页注意力机制在推理服务中的作用,要求提到 KV cache、页表和显存碎片。"
def gen(prompt, max_tokens):
    payload={"model":"qwen36","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,
             "temperature":0,"stream":True,"stream_options":{"include_usage":True}}
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); ttft=None; toks=[]
    with urllib.request.urlopen(req,timeout=900) as r:
        for raw in r:
            line=raw.decode().strip()
            if not line.startswith("data: "): continue
            body=line[6:]
            if body=="[DONE]": break
            ch=json.loads(body)
            if ch.get("usage"): continue
            d=(ch.get("choices") or [{}])[0].get("delta",{}).get("content")
            if d:
                now=time.perf_counter()
                if ttft is None: ttft=now-t0
                toks.append(now)
    wall=(toks[-1] if toks else time.perf_counter())-t0
    return {"ttft":ttft or wall,"tpot":(wall-(ttft or 0))/max(1,len(toks)-1),"tokens":len(toks)}
def pct(xs,p):
    xs=sorted(xs); k=max(0,min(len(xs)-1,int(round(p/100*(len(xs)-1))))); return round(xs[k]*1000,2)
gen(SHORT,16)
res=[]
for _ in range(RUNS):
    t0=time.perf_counter()
    with ThreadPoolExecutor(max_workers=C) as ex: rs=list(ex.map(lambda _: gen(SHORT,128), range(C)))
    wall=time.perf_counter()-t0; total=sum(r["tokens"] for r in rs)
    res.append((total/wall,[r["ttft"] for r in rs],[r["tpot"] for r in rs]))
o={"conc":C,"runs":RUNS,"out_toks_p50":round(statistics.median([x[0] for x in res]),2),
   "ttft_ms_p50":pct([v for x in res for v in x[1]],50),"tpot_ms_p50":pct([v for x in res for v in x[2]],50)}
print(json.dumps(o)); json.dump(o,open(OUT,'w'),indent=2)
