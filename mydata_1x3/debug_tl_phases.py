"""
Debug traffic-light phase cycling for CityFlow.

Run from repository root (same as run_data_1x3.py) so paths in config resolve:
  cd /app && python3 mydata_1x3/debug_tl_phases.py

Root cause of "frozen" phases:
  When rlTrafficLight is TRUE, Engine::nextStep() does NOT call passTime() on
  traffic lights. Phases only change if you call eng.set_tl_phase(...). So a
  script that never calls set_tl_phase (e.g. run_data_1x3.py) will stay on the
  initial phase forever.

  When rlTrafficLight is FALSE, fixed lightphases in the roadnet advance via
  passTime() each step.

Note: The stock CityFlow Python binding has set_tl_phase but NO get_tl_phase.
We infer phase changes from the replay log tail (road/lane g+r pattern).
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile


def _signal_tail(replay_line: str) -> str:
    if ";" not in replay_line:
        return ""
    return replay_line.split(";", 1)[1].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="CityFlow config JSON (default: mydata_1x3/config.json next to this script)",
    )
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(base, ".."))
    cfg_path = args.config or os.path.join(base, "config.json")
    cfg_path = os.path.abspath(cfg_path)

    with open(cfg_path, "r") as f:
        cfg_doc = json.load(f)

    # (1) Validate non-virtual intersections have trafficLight + multiple lightphases
    rn = cfg_doc.get("roadnetFile", "").lstrip("./").replace("\\", "/")
    roadnet_path = os.path.normpath(os.path.join(root, rn))
    if os.path.isfile(roadnet_path):
        with open(roadnet_path, "r") as f:
            rn_data = json.load(f)
        print("\n--- Roadnet traffic lights (non-virtual only) ---")
        for inter in rn_data.get("intersections", []):
            if inter.get("virtual"):
                continue
            tid = inter.get("id", "?")
            tl = inter.get("trafficLight") or {}
            phases = tl.get("lightphases") or []
            times = [p.get("time") for p in phases if isinstance(p, dict)]
            ok = "OK" if len(phases) > 1 else "WARNING: need 2+ phases to cycle"
            print(f"  {tid}: {len(phases)} phases, times (first 5)={times[:5]!r} {ok}")
    else:
        print(f"(Skipping roadnet validation; file not found: {roadnet_path})")

    rl = cfg_doc.get("rlTrafficLight")
    print(f"Config file: {cfg_path}")
    print(f"rlTrafficLight: {rl}")
    if rl is True:
        print(
            "\n*** WARNING: rlTrafficLight=true means fixed timers in the roadnet are NOT "
            "advanced by the engine. Phases stay at the initial index unless you call "
            "eng.set_tl_phase(intersection_id, phase_index) each step (RL mode).\n"
        )
    else:
        print(
            "\nrlTrafficLight=false: fixed lightphases should advance via passTime() each step.\n"
        )

    # Write a temp config so replay goes to a scratch file (do not clobber replay.txt).
    cfg_work = dict(cfg_doc)
    cfg_work["replayLogFile"] = "./mydata_1x3/_replay_debug_tl.txt"
    cfg_work["roadnetLogFile"] = "./mydata_1x3/_replay_roadnet_debug_tl.json"
    fd, tmp_cfg = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(tmp_cfg, "w") as f:
            json.dump(cfg_work, f, indent=2)
        print(f"Temp engine config: {tmp_cfg}")

        os.chdir(root)
        import cityflow

        eng = cityflow.Engine(os.path.abspath(tmp_cfg).replace("\\", "/"), thread_num=1)

        tails: dict[int, str] = {}
        for step in range(args.steps):
            eng.next_step()
            t = float(eng.get_current_time())
            if step in (0, 4, 5, 6, 29, 30, 31) or step == args.steps - 1:
                # get_tl_phase does not exist in Python; time == step * interval for basic runs
                print(f"step={step:4d}  get_current_time()={t:g}")

        # Read scratch replay and compare signal tails (proves g/r pattern changes if phases advance).
        scratch = os.path.join(root, "mydata_1x3", "_replay_debug_tl.txt")
        if not os.path.isfile(scratch):
            print("No scratch replay written; check cwd and config paths.")
            return
        with open(scratch, "r") as f:
            lines = [f.readline() for _ in range(min(args.steps, 500))]

        for idx in [0, 4, 5, 6, 29, 30, 31, min(100, len(lines) - 1)]:
            if idx < len(lines) and lines[idx]:
                tails[idx] = _signal_tail(lines[idx])

        unique = len(set(tails.values()))
        print("\n--- Replay signal-tail comparison (after ';') ---")
        for k in sorted(tails):
            h = hash(tails[k]) & 0xFFFFFFFFFFFFFFFF
            print(f"line {k}: hash={h}  preview={tails[k][:100]!r}...")

        if unique <= 1 and len(tails) > 1:
            print(
                "\n*** All sampled replay tails are identical: lane g/r did not change. "
                "If rlTrafficLight is false, this is unexpected — check engine build. "
                "If rlTrafficLight is true, this is expected without set_tl_phase()."
            )
        elif unique > 1:
            print("\nReplay tails differ: traffic signal state is changing over time.")
    finally:
        try:
            os.remove(tmp_cfg)
        except OSError:
            pass
        for p in (
            os.path.join(root, "mydata_1x3", "_replay_debug_tl.txt"),
            os.path.join(root, "mydata_1x3", "_replay_roadnet_debug_tl.json"),
        ):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()
