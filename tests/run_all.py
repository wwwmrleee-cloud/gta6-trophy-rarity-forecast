"""Run the spec's tests 1-5 in one go. Exit 0 only if every check passes.

Usage: python3 tests/run_all.py  (run from the repository root)
Prints every measured number; the README quotes the synthetic ones.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = Path(os.environ.get(
    "KIT_DIR",
    "/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
    "gta6-trophy-rarity-forecast-01-kit"))
sys.path.insert(0, str(ROOT / "tests"))

from scorer import score, validate_output, load_out  # noqa: E402
from simulate import MORNING, build_case  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def run_prog(script, in_dir, out_dir, env_extra=None):
    env = dict(os.environ)
    env["MAKEFUN_INPUT_DIR"] = str(in_dir)
    env["MAKEFUN_OUTPUT_DIR"] = str(out_dir)
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(script)], env=env,
                          capture_output=True, text=True, timeout=120)
    return proc, time.time() - t0


def main():
    tmp = Path(tempfile.mkdtemp(prefix="gta6-tests-"))
    try:
        # ---- Test 1: scorer reproduces the worked examples. ----
        forecast = {"trophies": [{"id": 1, "q": [10, 15, 20, 25, 35]},
                                 {"id": 2, "q": [10, 15, 20, 25, 35]}]}
        loss, miss, cov = score(forecast, {1: 22.0, 2: 8.0})
        check("test1 worked example loss", abs(loss - 0.403404) < 1e-6,
              f"loss={loss:.6f} (want 0.403404)")
        check("test1 typical miss", abs(miss - 7.0) < 1e-9,
              f"miss={miss}")
        check("test1 coverage", abs(cov - 0.5) < 1e-9, f"cov={cov}")
        invalid = {"trophies": [{"id": 1, "q": [30] * 5},
                                {"id": 2, "q": [5] * 5}]}
        loss2, _, _ = score(invalid, {1: 22.0, 2: 8.0})
        check("test1 invalid-case loss", abs(loss2 - 0.46023) < 1e-6,
              f"loss={loss2:.6f} (want 0.46023)")

        # ---- Test 2: practice cases, valid + byte-identical. ----
        kit_in = KIT / "input"
        out_a, out_b = tmp / "out_a", tmp / "out_b"
        proc_a, dt_a = run_prog(ROOT / "main.py", kit_in, out_a)
        proc_b, dt_b = run_prog(ROOT / "main.py", kit_in, out_b)
        check("test2 run p00/p01 clean",
              proc_a.returncode == 0 and proc_b.returncode == 0,
              (proc_a.stderr + proc_b.stderr).strip()[:300])
        for case_id in ("p00", "p01"):
            fa, fb = out_a / f"{case_id}.out", out_b / f"{case_id}.out"
            data_a, data_b = fa.read_bytes(), fb.read_bytes()
            check(f"test2 {case_id} byte-identical", data_a == data_b)
            check(f"test2 {case_id} under 65536 bytes",
                  len(data_a) <= 65536, f"{len(data_a)} bytes")
            case = json.loads((kit_in / f"{case_id}.in").read_text())
            ids = [t["id"] for t in case["trophies"]]
            check(f"test2 {case_id} valid shape",
                  validate_output(load_out(fa), ids), f"{len(ids)} rows")
        print(f"test2 runtimes: {dt_a:.2f}s and {dt_b:.2f}s for both cases")

        # ---- Test 3: baselines on the practice, 09:37 as stand-in answer.
        p00 = json.loads((kit_in / "p00.in").read_text())
        synth3 = dict(p00)
        synth3["id"] = "s03"
        synth3["at"] = "2026-11-19T03:00:00Z"
        synth3["target"] = "2026-11-19T09:37:00Z"
        synth3["horizon_minutes"] = 397
        case3_dir = tmp / "case3_in"
        case3_dir.mkdir()
        (case3_dir / "s03.in").write_text(json.dumps(synth3))
        truth3 = {i + 1: MORNING[i] for i in range(55)}
        results3 = {}
        for name, script in (("entry", ROOT / "main.py"),
                             ("starter", KIT / "starter" / "main.py"),
                             ("trend-only", ROOT / "tests" / "trend_only.py")):
            out_dir = tmp / f"out3_{name}"
            proc, _ = run_prog(script, case3_dir, out_dir)
            if proc.returncode != 0:
                check(f"test3 {name} runs", False, proc.stderr[:200])
                continue
            loss, miss, cov = score(load_out(out_dir / "s03.out"), truth3)
            results3[name] = (loss, miss, cov)
            print(f"test3 {name}: loss={loss:.6f} miss={miss:.4f} cov={cov:.3f}")
        check("test3 all three scored", len(results3) == 3)

        # ---- Test 4: synthetic final-like case. ----
        sim_in = tmp / "sim_in"
        sim_in.mkdir()
        _, truth4 = build_case(str(kit_in / "p00.in"), str(sim_in / "f00.in"))
        results4 = {}
        for name, script in (("entry", ROOT / "main.py"),
                             ("starter", KIT / "starter" / "main.py"),
                             ("trend-only", ROOT / "tests" / "trend_only.py")):
            out_dir = tmp / f"out4_{name}"
            proc, dt = run_prog(script, sim_in, out_dir)
            if proc.returncode != 0:
                check(f"test4 {name} runs", False, proc.stderr[:300])
                continue
            check(f"test4 {name} fast", dt < 8.0, f"{dt:.2f}s")
            loss, miss, cov = score(load_out(out_dir / "f00.out"), truth4)
            results4[name] = (loss, miss, cov)
            print(f"test4 {name}: loss={loss:.6f} miss={miss:.4f} "
                  f"cov={cov:.3f}")
        if len(results4) == 3:
            e, s, t = (results4[k][0] for k in ("entry", "starter",
                                                "trend-only"))
            check("test4 entry beats starter", e < s, f"{e:.6f} < {s:.6f}")
            check("test4 entry beats trend-only", e < t,
                  f"{e:.6f} < {t:.6f}")
            cov4 = results4["entry"][2]
            check("test4 entry coverage 0.7-1.0", 0.7 <= cov4 <= 1.0,
                  f"cov={cov4:.3f}")
        else:
            check("test4 all three scored", False)

        # ---- Test 5: degenerate cases. ----
        base = json.loads((kit_in / "p00.in").read_text())
        trophies = base["trophies"]

        def case_with(cid, history, trophies_list=None, horizon=90,
                      target="2026-11-19T04:30:00Z",
                      at="2026-11-19T03:00:00Z"):
            return {"schema": "makefun/gta6-trophies-case/v1", "id": cid,
                    "at": at, "horizon_minutes": horizon, "target": target,
                    "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
                    "game": base["game"],
                    "trophies": trophies_list or trophies,
                    "history": history, "notes": "degenerate test case"}

        first = base["history"][0]
        flat = [dict(first, at=first["at"])] * 5
        rates100 = list(first["rates"])
        rates100[2] = 100.0
        rates0 = list(first["rates"])
        rates0[3] = 0.0
        missing = [dict(first, rates=[None] * 55),
                   dict(first, rates=list(first["rates"]))]
        small_trophies = trophies[:3]
        small_hist = [dict(first, rates=list(first["rates"])[:3])]
        deg_dir = tmp / "deg_in"
        deg_dir.mkdir()
        cases = {
            "d01": case_with("d01", [first]),
            "d02": case_with("d02", flat),
            "d03": case_with("d03", [dict(first, rates=rates100),
                                     dict(first, rates=rates100)]),
            "d04": case_with("d04", [dict(first, rates=rates0),
                                     dict(first, rates=rates0)]),
            "d05": case_with("d05", missing),
            "d06": case_with("d06", base["history"], horizon=90),
            "d07": case_with("d07", base["history"], horizon=6510,
                             target="2026-11-26T00:00:00Z"),
            "d08": case_with("d08", small_hist, trophies_list=small_trophies),
            "d09": case_with("d09", []),
        }
        for cid, case in cases.items():
            (deg_dir / f"{cid}.in").write_text(json.dumps(case))
        out_deg = tmp / "out_deg"
        proc, dt = run_prog(ROOT / "main.py", deg_dir, out_deg)
        check("test5 degenerate run clean", proc.returncode == 0,
              proc.stderr[:300])
        check("test5 degenerate fast", dt < 8.0, f"{dt:.2f}s")
        for cid, case in cases.items():
            path = out_deg / f"{cid}.out"
            ok = path.exists()
            if ok:
                ids = [t["id"] for t in case["trophies"]]
                try:
                    validate_output(load_out(path), ids)
                except AssertionError as exc:
                    ok = False
                    print(f"  {cid} invalid: {exc}")
            check(f"test5 {cid} valid", ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
