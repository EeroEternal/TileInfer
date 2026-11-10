#!/usr/bin/env python3
"""ti_trace.py — 统一的性能监测入口（只有一个脚本、一份配置、一个输出目录）。

它把原来散落的监测脚本收进来，作为"预设(preset)"：

    perop.sh / perop_split.sh / prefill_timing.sh / stepshare_ab.sh / tpot_leaf.sh

常用命令：

    python3 scripts/ti_trace.py presets                 # 列出所有预设
    python3 scripts/ti_trace.py config                  # 当前生效配置（含来源）
    python3 scripts/ti_trace.py level L2                # 切换检测级别（写配置）
    python3 scripts/ti_trace.py run --preset perop_split --level L2
    python3 scripts/ti_trace.py run --preset smoke --dry-run
    python3 scripts/ti_trace.py report trace_out/latest

报告目录：<repo>/trace_out/<时间戳>-<preset>/ ，并维护 trace_out/latest 软链。

级别（在哪里配置）：
    1) 本文件同目录的 ti_trace.json（项目配置，优先）
    2) ~/.tileinfer/ti_trace.json（个人配置）
    3) 命令行 --level 覆盖
    三个来源的优先级：命令行 > 项目配置 > 个人配置 > 内置默认。
    级别只决定"观测强度"，全部写进上面那份 JSON 的 levels 段。

退出码：0 成功；2 设备忙/配置错误；3 服务未就绪；4 压测失败。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
PROJECT_CFG = Path(__file__).resolve().with_name("ti_trace.json")
USER_CFG = Path.home() / ".tileinfer" / "ti_trace.json"
OUT_ROOT = REPO / "trace_out"
LATEST_LINK = OUT_ROOT / "latest"
LOCK = Path("/tmp/tileinfer_dev.lock")

# 内置默认：即使没有任何配置文件也能跑
DEFAULTS: Dict[str, Any] = {
    "python": "/home/userlist/zxs/TileInfer/.venv_native213/bin/python",
    "model": "/home/models/Qwen3.6-35B-A3B",
    "served_name": "qwen36",
    "port": 8111,
    "device_mem_limit_mb": 20000,
    "env": {
        "TILELANG_CACHE_DIR": "/home/userlist/zxs/tl_cache_213",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_DO_NOT_TRACK": "1",
    },
    "serve_args": [
        "--attention-backend", "CUSTOM",
        "--enforce-eager",
        "--max-model-len", "9216",
        "--max-num-batched-tokens", "9216",
        "--max-num-seqs", "32",
        "--gpu-memory-utilization", "0.75",
        "--trust-remote-code",
    ],
    "levels": {
        "L0": {
            "desc": "关闭观测：正式性能验收必须用这一档",
            "env": {"TILEINFER_PHASE_TIMING": "0"},
        },
        "L1": {
            "desc": "阶段级（host 时钟），开销 <1%",
            "env": {"TILEINFER_PHASE_TIMING": "host"},
        },
        "L2": {
            "desc": "算子级（host+device），开销 2-5%，默认",
            "env": {
                "TILEINFER_PHASE_TIMING": "device",
                "TILEINFER_PHASE_OPS": "detail",
                "TILEINFER_PHASE_LEAF_LIMIT": "0",
            },
        },
        "L3": {
            "desc": "算子级 + 单次前向全量（开销集中在被采样的那一步）",
            "env": {
                "TILEINFER_PHASE_TIMING": "both",
                "TILEINFER_PHASE_OPS": "detail",
                "TILEINFER_PHASE_LEAF_LIMIT": "0",
                "TILEINFER_TRACE_FORWARD": "1",
            },
        },
    },
    "presets": {
        "smoke": {
            "desc": "冒烟：单请求短输入，~40 秒，确认服务与算子正常",
            "level": "L2",
            "scenarios": [{"name": "smoke", "prompt": 128, "out": 32, "conc": 1, "repeat": 1}],
        },
        "perop": {
            "desc": "等价 perop.sh：512-token 预填充的逐算子表",
            "level": "L2",
            "scenarios": [{"name": "prefill_512", "prompt": 512, "out": 8, "conc": 1, "repeat": 1}],
        },
        "perop_split": {
            "desc": "等价 perop_split.sh：拆到 GDN 函数级",
            "level": "L2",
            "scenarios": [{"name": "prefill_512", "prompt": 512, "out": 8, "conc": 1, "repeat": 1}],
        },
        "prefill_timing": {
            "desc": "等价 prefill_timing.sh：预填充分段计时",
            "level": "L2",
            "scenarios": [
                {"name": "prefill_512", "prompt": 512, "out": 4, "conc": 1, "repeat": 1},
                {"name": "prefill_2k", "prompt": 2048, "out": 4, "conc": 1, "repeat": 1},
            ],
        },
        "stepshare": {
            "desc": "等价 stepshare_ab.sh：步内占比",
            "level": "L2",
            "scenarios": [{"name": "decode_128", "prompt": 128, "out": 128, "conc": 1, "repeat": 1}],
        },
        "tpot_leaf": {
            "desc": "等价 tpot_leaf.sh：decode 叶子耗时",
            "level": "L3",
            "scenarios": [{"name": "decode_128", "prompt": 128, "out": 96, "conc": 1, "repeat": 2}],
        },
        "matrix_s": {
            "desc": "§6 达标组四场景（S1-S4）",
            "level": "L2",
            "scenarios": [
                {"name": "S1_short_c1", "prompt": 128, "out": 128, "conc": 1, "repeat": 2},
                {"name": "S2_4k_c1", "prompt": 4096, "out": 128, "conc": 1, "repeat": 1},
                {"name": "S3_short_c16", "prompt": 128, "out": 128, "conc": 16, "repeat": 1},
                {"name": "S4_2k_c8", "prompt": 2048, "out": 128, "conc": 8, "repeat": 1},
            ],
        },
    },
}


# ---------------------------------------------------------------- 配置


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> tuple[Dict[str, Any], List[str]]:
    """返回 (配置, 生效来源列表)；后者用于 `config` 命令展示。"""
    cfg, src = dict(DEFAULTS), ["builtin"]
    for path, tag in ((USER_CFG, "user"), (PROJECT_CFG, "project")):
        if path.is_file():
            try:
                cfg = _merge(cfg, json.loads(path.read_text(encoding="utf-8")))
                src.append(f"{tag}:{path}")
            except Exception as e:  # 配置坏了要明说，不能静默
                print(f"[warn] 读取配置失败 {path}: {e}", file=sys.stderr)
    return cfg, src


def save_level(level: str) -> None:
    data = {}
    if PROJECT_CFG.is_file():
        data = json.loads(PROJECT_CFG.read_text(encoding="utf-8"))
    data["level"] = level
    PROJECT_CFG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 设备


def npu_used_mb() -> Optional[int]:
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    m = re.search(r"(\d+)\s*/\s*131072", out)
    return int(m.group(1)) if m else None


class DeviceLock:
    """单卡排队：拿不到锁就不跑，避免三个人互相踩。"""

    def __init__(self, owner: str):
        self.owner, self.held = owner, False

    def __enter__(self):
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{self.owner} {os.getpid()} {time.time()}\n".encode())
            os.close(fd)
            self.held = True
        except FileExistsError:
            txt = LOCK.read_text(errors="replace").strip()
            raise SystemExit(f"[busy] 设备锁已被占用：{txt}\n  若确认对方已结束，删掉 {LOCK} 再试。")
        return self

    def __exit__(self, *a):
        if self.held:
            try:
                LOCK.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------- 压测客户端


def _one_request(port: int, model: str, prompt: str, max_tokens: int, timeout: float) -> Dict[str, float]:
    """一次流式请求，返回 TTFT / 每 token 间隔 / 输出 token 数。"""
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    stamps: List[float] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("choices") and obj["choices"][0].get("text"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = (now - t0) * 1000.0
                stamps.append(now)
    total = (time.perf_counter() - t0) * 1000.0
    gaps = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
    return {
        "ttft_ms": ttft if ttft is not None else total,
        "total_ms": total,
        "n_out": max(1, len(stamps)),
        "tpot_ms": (sum(gaps) / len(gaps)) if gaps else total,
    }


def _pct(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def run_scenario(port: int, model: str, sc: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    prompt = "hello " * max(1, sc["prompt"] // 2)  # 粗略按 token 数铺长度
    results: List[Dict[str, float]] = []
    errors: List[str] = []
    threads: List[threading.Thread] = []
    lock = threading.Lock()

    def worker():
        for _ in range(sc.get("repeat", 1)):
            try:
                r = _one_request(port, model, prompt, sc["out"], timeout)
                with lock:
                    results.append(r)
            except Exception as e:
                with lock:
                    errors.append(f"{type(e).__name__}: {e}")

    t0 = time.perf_counter()
    for _ in range(sc["conc"]):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    out_tokens = sum(r["n_out"] for r in results)
    return {
        "name": sc["name"],
        "prompt": sc["prompt"],
        "conc": sc["conc"],
        "ok": len(results),
        "errors": errors[:5],
        "ttft_p50": _pct([r["ttft_ms"] for r in results], 0.50),
        "ttft_p90": _pct([r["ttft_ms"] for r in results], 0.90),
        "tpot_p50": _pct([r["tpot_ms"] for r in results], 0.50),
        "tpot_p90": _pct([r["tpot_ms"] for r in results], 0.90),
        "tpot_max": max([r["tpot_ms"] for r in results], default=0.0),
        "tok_s": out_tokens / wall if wall > 0 else 0.0,
        "req_s": len(results) / wall if wall > 0 else 0.0,
    }


# ---------------------------------------------------------------- 报告


PHASE_ROW = re.compile(r"^\s*(op:[a-z_]+:[a-zA-Z0-9_]+|stage:[a-z_]+)\s+(.+)$")


def build_report(outdir: Path, plan: Dict[str, Any], scen: List[Dict[str, Any]]) -> Path:
    raw = outdir / "raw"
    serve_log = raw / "serve.log"
    tables: List[str] = []
    if serve_log.is_file():
        text = serve_log.read_text(encoding="utf-8", errors="replace")
        # 抓服务日志里最后几段相位表（表格以表头行开始，遇到空行终止）
        for m in re.finditer(r"(?m)^(phase\s+.*(?:\n.*)*?)(?:\n\s*\n)", text):
            block = m.group(1).rstrip()
            if "dev" in block or "host" in block:
                tables.append(block)
    tail = tables[-1] if tables else "(服务日志里没有相位表：确认级别 >= L1，且运行时间足够)"

    summary = outdir / "summary.md"
    lines = [
        f"# 性能报告 {plan['stamp']}",
        "",
        f"- 预设：`{plan['preset']}`  级别：`{plan['level']}`（{plan['level_desc']}）",
        f"- 模型：`{plan['model']}`  端口：{plan['port']}",
        f"- 启动参数：`{' '.join(plan['serve_args'])}`",
        f"- 设备起始占用：{plan.get('npu_used_mb_before', '?')} MB",
        "",
        "## 1. 场景结果",
        "",
        "| 场景 | 输入 | 并发 | 成功/失败 | TTFT p50/p90 (ms) | TPOT p50/p90/max (ms) | 输出 tok/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in scen:
        lines.append(
            f"| {s['name']} | {s['prompt']} | {s['conc']} | {s['ok']}/{len(s['errors'])} | "
            f"{s['ttft_p50']:.1f} / {s['ttft_p90']:.1f} | "
            f"{s['tpot_p50']:.2f} / {s['tpot_p90']:.2f} / {s['tpot_max']:.2f} | {s['tok_s']:.2f} |"
        )
    lines += ["", "## 2. 服务侧相位/算子表（最后一段）", "", "```", tail, "```", ""]
    lines += [
        "## 3. 尚未实现（Phase 2）",
        "",
        "- 异常检测（R1–R6）与异常置顶：目前只有原始表，异常不会被自动标出",
        "- `ops.csv` / `layers.csv` / `anomalies.csv` / `trace.json`：需要更细的插桩",
        "- 单次前向全量报告 `forward_<n>.md`：需要 `TILEINFER_TRACE_FORWARD` 落地",
        "",
        f"> 原始日志：`{raw}`",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")

    (outdir / "meta.json").write_text(json.dumps({
        "stamp": plan["stamp"], "preset": plan["preset"], "level": plan["level"],
        "level_desc": plan["level_desc"], "serve_args": plan["serve_args"],
        "env": plan["env"], "model": plan["model"], "port": plan["port"],
        "npu_used_mb_before": plan.get("npu_used_mb_before"),
        "scenarios": scen,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (outdir / "stages.csv").open("w", encoding="utf-8") as f:
        f.write("scenario,prompt,conc,ttft_p50_ms,ttft_p90_ms,tpot_p50_ms,tpot_p90_ms,tpot_max_ms,tok_s,req_s\n")
        for s in scen:
            f.write(f"{s['name']},{s['prompt']},{s['conc']},{s['ttft_p50']:.2f},{s['ttft_p90']:.2f},"
                    f"{s['tpot_p50']:.3f},{s['tpot_p90']:.3f},{s['tpot_max']:.3f},{s['tok_s']:.3f},{s['req_s']:.3f}\n")
    return summary


# ---------------------------------------------------------------- 命令


def cmd_presets(cfg, src, args):
    print("可用预设：\n")
    for name, p in cfg["presets"].items():
        n = len(p.get("scenarios", []))
        print(f"  {name:<14} level={p.get('level','L2'):<3} {n} 个场景  {p['desc']}")
    print("\n可用级别：\n")
    for name, lv in cfg["levels"].items():
        print(f"  {name:<3} {lv['desc']}")
        print(f"      env: {lv['env']}")
    return 0


def cmd_config(cfg, src, args):
    print(f"生效来源：{', '.join(src)}")
    print(f"项目配置：{PROJECT_CFG}  {'(存在)' if PROJECT_CFG.is_file() else '(不存在)'}")
    print(f"个人配置：{USER_CFG}  {'(存在)' if USER_CFG.is_file() else '(不存在)'}")
    print(f"输出根目录：{OUT_ROOT}")
    cur = cfg.get("level", "L2")
    print(f"当前默认级别：{cur} — {cfg['levels'].get(cur, {}).get('desc', '?')}")
    if args.json:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
    return 0


def cmd_level(cfg, src, args):
    if args.name not in cfg["levels"]:
        print(f"[error] 未知级别 {args.name}；可选：{', '.join(cfg['levels'])}", file=sys.stderr)
        return 2
    save_level(args.name)
    print(f"已把默认级别写为 {args.name}（{cfg['levels'][args.name]['desc']}）→ {PROJECT_CFG}")
    return 0


def cmd_report(cfg, src, args):
    d = Path(args.dir)
    if not d.is_absolute():
        d = (REPO / d).resolve()
    if d.name == "latest" and d.is_symlink():
        d = d.resolve()
    summary = d / "summary.md"
    if summary.is_file():
        print(summary.read_text(encoding="utf-8"))
    else:
        print(f"[error] {d} 下没有 summary.md", file=sys.stderr)
        return 2
    return 0


def cmd_run(cfg, src, args):
    preset_name = args.preset
    if preset_name not in cfg["presets"]:
        print(f"[error] 未知预设 {preset_name}；用 `presets` 查看", file=sys.stderr)
        return 2
    preset = cfg["presets"][preset_name]
    level = args.level or preset.get("level") or cfg.get("level", "L2")
    if level not in cfg["levels"]:
        print(f"[error] 未知级别 {level}", file=sys.stderr)
        return 2

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = OUT_ROOT / f"{stamp}-{preset_name}"
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg["env"].items()})
    env.update({k: str(v) for k, v in cfg["levels"][level]["env"].items()})

    used = npu_used_mb()
    plan = {
        "stamp": stamp, "preset": preset_name, "level": level,
        "level_desc": cfg["levels"][level]["desc"],
        "serve_args": list(cfg["serve_args"]), "env": {k: str(v) for k, v in cfg["levels"][level]["env"].items()},
        "model": cfg["model"], "port": cfg["port"], "npu_used_mb_before": used,
    }

    print(f"[plan] 预设={preset_name} 级别={level} 输出={outdir}")
    print(f"[plan] 场景数={len(preset['scenarios'])} 设备占用={used} MB")
    print(f"[plan] 观测环境变量={plan['env']}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[plan] 报告目录：{outdir}")
    if args.dry_run:
        print("[dry-run] 不启动服务、不压测；已写出 plan.json。")
        return 0

    if used is not None and used > cfg["device_mem_limit_mb"] and not args.force:
        print(f"[busy] 设备已占用 {used} MB > 门限 {cfg['device_mem_limit_mb']} MB；"
              f"用 --force 跳过检查。", file=sys.stderr)
        return 2

    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    with DeviceLock(f"ti_trace:{preset_name}"):
        cmd = [cfg["python"], "-m", "vllm.entrypoints.cli.main", "serve", cfg["model"],
               "--served-model-name", cfg["served_name"], "--port", str(cfg["port"])] + list(cfg["serve_args"])
        print(f"[serve] {' '.join(cmd)}")
        log = (outdir / "raw" / "serve.log").open("wb")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO))
        try:
            ready, waited = False, 0.0
            while waited < args.startup_timeout:
                if proc.poll() is not None:
                    print(f"[error] 服务进程提前退出，见 {outdir/'raw'/'serve.log'}", file=sys.stderr)
                    return 3
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{cfg['port']}/v1/models", timeout=3).read()
                    ready = True
                    break
                except Exception:
                    time.sleep(3)
                    waited += 3
            if not ready:
                print(f"[error] {args.startup_timeout}s 内服务未就绪", file=sys.stderr)
                return 3
            print(f"[serve] 就绪（{waited:.0f}s），开始压测")
            scen = []
            for sc in preset["scenarios"]:
                r = run_scenario(cfg["port"], cfg["served_name"], sc, args.request_timeout)
                scen.append(r)
                print(f"  {r['name']:<14} ok={r['ok']} TTFT {r['ttft_p50']:.1f}ms "
                      f"TPOT {r['tpot_p50']:.2f}ms {r['tok_s']:.2f} tok/s")
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()

    summary = build_report(outdir, plan, scen)
    (OUT_ROOT / "latest").unlink(missing_ok=True)
    os.symlink(outdir, OUT_ROOT / "latest")
    print(f"\n[report] {summary}")
    print(f"[report] 目录：{outdir}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="ti_trace.py", description="TileInfer 统一性能监测入口")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("presets", help="列出预设与级别")
    sp.set_defaults(func=cmd_presets)

    sp = sub.add_parser("config", help="显示生效配置")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_config)

    sp = sub.add_parser("level", help="切换默认级别（写入项目配置）")
    sp.add_argument("name", help="L0 / L1 / L2 / L3")
    sp.set_defaults(func=cmd_level)

    sp = sub.add_parser("report", help="从已有目录重新打印报告")
    sp.add_argument("dir", nargs="?", default="trace_out/latest")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("run", help="一键：起服务 → 压测 → 出报告")
    sp.add_argument("--preset", default="smoke")
    sp.add_argument("--level", default=None)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--force", action="store_true", help="跳过设备显存门")
    sp.add_argument("--startup-timeout", type=float, default=600.0)
    sp.add_argument("--request-timeout", type=float, default=1800.0)
    sp.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    cfg, src = load_config()
    return args.func(cfg, src, args)


if __name__ == "__main__":
    raise SystemExit(main())
