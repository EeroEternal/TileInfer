"""Long-prompt TTFT/TPOT probe, streaming, median of N runs.

Kept in-repo so the numbers in docs/notes/m1-v3-bridge.md can be re-derived after
/tmp is cleaned.  Usage:

    TI_PROMPT=/tmp/prompt-8k.txt python benchmarks/v3_long_ttft.py <port> <label> <out.json> [runs]

`TI_PROMPT` defaults to the 3.9K prompt used by M1; `TI_MAX_TOKENS` defaults to 32.
Build prompts with `benchmarks/make_long_prompt.py --tokens N --out ...`.
"""

import json, os, statistics, sys, time, urllib.request
PORT=int(sys.argv[1]); LABEL=sys.argv[2]; OUT=sys.argv[3]; RUNS=int(sys.argv[4]) if len(sys.argv)>4 else 3
PROMPT_PATH=os.environ.get("TI_PROMPT","/tmp/long4k_prompt.txt")
MAX_TOKENS=int(os.environ.get("TI_MAX_TOKENS","32"))
PROMPT=open(PROMPT_PATH,encoding='utf-8').read()
BASE=f"http://127.0.0.1:{PORT}"

def one():
    payload={"model":"qwen36","messages":[{"role":"user","content":PROMPT}],"max_tokens":MAX_TOKENS,
             "temperature":0,"stream":True,"stream_options":{"include_usage":True}}
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); ttft=None; toks=[]; pt=None
    with urllib.request.urlopen(req,timeout=900) as r:
        for raw in r:
            line=raw.decode().strip()
            if not line.startswith("data: "): continue
            body=line[6:]
            if body=="[DONE]": break
            ch=json.loads(body)
            if ch.get("usage"): pt=ch["usage"].get("prompt_tokens"); continue
            d=(ch.get("choices") or [{}])[0].get("delta",{}).get("content")
            if d:
                now=time.perf_counter()
                if ttft is None: ttft=now-t0
                toks.append(now)
    wall=(toks[-1] if toks else time.perf_counter())-t0
    return {"ttft":ttft or wall,"tpot":(wall-(ttft or 0))/max(1,len(toks)-1),"tokens":len(toks),"ptoks":pt}

one()  # warm
res=[one() for _ in range(RUNS)]
o={"label":LABEL,"port":PORT,"runs":RUNS,"prompt_tokens":res[0]["ptoks"],
   "prompt":PROMPT_PATH,"max_tokens":MAX_TOKENS,
   "ttft_ms_p50":round(statistics.median([r["ttft"] for r in res])*1000,2),
   "tpot_ms_p50":round(statistics.median([r["tpot"] for r in res])*1000,3),
   "out_toks_p50":round(statistics.median([r["tokens"] for r in res]),2)}
print(json.dumps(o,ensure_ascii=False)); json.dump(o,open(OUT,'w'),ensure_ascii=False,indent=2)
