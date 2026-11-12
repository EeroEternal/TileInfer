"""Streaming A/B: single-stream TTFT/TPOT and concurrency throughput, 3-5 runs, median.

Kept in-repo so the numbers in docs/notes/m1-v3-bridge.md can be re-derived after
/tmp is cleaned.  Usage: python benchmarks/v3_ab.py <port> <label> <out.json> [runs]
"""

import json, statistics, time, urllib.request, sys
from concurrent.futures import ThreadPoolExecutor

PORT=int(sys.argv[1]); LABEL=sys.argv[2]; OUT=sys.argv[3]; RUNS=int(sys.argv[4]) if len(sys.argv)>4 else 2
BASE=f"http://127.0.0.1:{PORT}"; MODEL="qwen36"
SHORT="用三句话介绍分页注意力机制在推理服务中的作用,要求提到 KV cache、页表和显存碎片。"
LONG=("背景材料:分页注意力机制把 KV cache 划分为固定大小的页,通过页表把逻辑位置映射到"
"物理页,从而消除预分配连续显存带来的碎片问题;在解码阶段,内核按页表索引读取当前"
"序列涉及的每一个物理页,并对末页做长度掩码;调度器在页不足时触发扩容或逐出。")*7+"\n请概括材料中反复出现的核心思想,一句话即可。"

def gen(prompt, max_tokens):
    payload={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,
             "temperature":0,"stream":True,"stream_options":{"include_usage":True}}
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); ttft=None; toks=[]; ptoks=None
    with urllib.request.urlopen(req,timeout=900) as r:
        for raw in r:
            line=raw.decode().strip()
            if not line.startswith("data: "): continue
            body=line[6:]
            if body=="[DONE]": break
            ch=json.loads(body)
            if ch.get("usage"): ptoks=(ch["usage"] or {}).get("prompt_tokens"); continue
            d=(ch.get("choices") or [{}])[0].get("delta",{}).get("content")
            if d:
                now=time.perf_counter()
                if ttft is None: ttft=now-t0
                toks.append(now)
    wall=(toks[-1] if toks else time.perf_counter())-t0
    tpot=(wall-(ttft or 0))/max(1,len(toks)-1)
    return {"ttft":ttft or wall,"tpot":tpot,"tokens":len(toks),"wall":wall,"ptoks":ptoks}

def pct(xs,p):
    xs=sorted(xs); k=max(0,min(len(xs)-1,int(round(p/100*(len(xs)-1))))); return round(xs[k]*1000,2)

def single(name,prompt,mt):
    ts=[]; tp=[]; pt=None
    for _ in range(RUNS):
        r=gen(prompt,mt); ts.append(r["ttft"]); tp.append(r["tpot"]); pt=r["ptoks"] or pt; time.sleep(0.3)
    o={"scenario":name,"prompt_tokens":pt,"max_tokens":mt,"ttft_ms_p50":pct(ts,50),"tpot_ms_p50":pct(tp,50),
       "out_toks_p50":round(statistics.median([1/x for x in tp if x>0])*1000,2) if any(x>0 for x in tp) else None}
    print(json.dumps(o,ensure_ascii=False),flush=True); return o

def conc(c,mt=128):
    res=[]
    for _ in range(RUNS):
        t0=time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            rs=list(ex.map(lambda _: gen(SHORT,mt), range(c)))
        wall=time.perf_counter()-t0; total=sum(r["tokens"] for r in rs)
        res.append((total/wall, c/wall, [r["ttft"] for r in rs], [r["tpot"] for r in rs]))
    tp=[x[0] for x in res]; rp=[x[1] for x in res]; tt=[v for x in res for v in x[2]]; tpt=[v for x in res for v in x[3]]
    o={"scenario":f"conc_{c}","conc":c,"max_tokens":mt,"out_toks_p50":round(statistics.median(tp),2),
       "req_toks_p50":round(statistics.median(rp),2),"ttft_ms_p50":pct(tt,50),"tpot_ms_p50":pct(tpt,50)}
    print(json.dumps(o,ensure_ascii=False),flush=True); return o

# warmup
gen(SHORT,16); gen(LONG,8); conc(4,16)
rep={"label":LABEL,"port":PORT,"runs":RUNS,"scenarios":[]}
for name,p,mt in [("ttft_short_1tok",SHORT,1),("decode_128tok",SHORT,128),("decode_256tok",SHORT,256),("long4k_128tok",LONG,128)]:
    rep["scenarios"].append(single(name,p,mt))
for c in (4,16):
    rep["scenarios"].append(conc(c,128))
json.dump(rep,open(OUT,"w"),ensure_ascii=False,indent=2)
print("SAVED",OUT)
