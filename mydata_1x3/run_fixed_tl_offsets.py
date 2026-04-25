import argparse
import json
import os
from typing import Dict, List

import cityflow


def _load_phase_times(roadnet_path: str, intersection_id: str) -> List[float]:
    with open(roadnet_path, "r") as f:
        rn = json.load(f)
    for inter in rn.get("intersections", []):
        if inter.get("id") != intersection_id:
            continue
        if inter.get("virtual"):
            raise ValueError(f"{intersection_id} is virtual in roadnet")
        phases = (inter.get("trafficLight") or {}).get("lightphases") or []
        times = [float(p.get("time", 0.0)) for p in phases if isinstance(p, dict)]
        if len(times) < 2:
            raise ValueError(f"{intersection_id} has <2 phases in roadnet")
        if any(t <= 0 for t in times):
            raise ValueError(f"{intersection_id} has non-positive phase times: {times}")
        return times
    raise KeyError(f"intersection id not found in roadnet: {intersection_id}")


def _phase_at_time(phase_times: List[float], t: float) -> int:
    cycle = sum(phase_times)
    if cycle <= 0:
        return 0
    x = t % cycle
    acc = 0.0
    for i, dt in enumerate(phase_times):
        acc += dt
        if x < acc:
            return i
    return len(phase_times) - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="./mydata_1x3/config_rl.json")
    ap.add_argument("--roadnet", type=str, default="./mydata_1x3/roadnet_1X3.json")
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--thread_num", type=int, default=1)

    # Offsets (seconds) to desync controllers.
    ap.add_argument("--offset_1_1", type=float, default=0.0)
    ap.add_argument("--offset_2_1", type=float, default=20.0)
    ap.add_argument("--offset_3_1", type=float, default=40.0)

    # Output replay path MUST be relative-ish (CityFlow prepends config dir).
    ap.add_argument("--replay_out", type=str, default="./mydata_1x3/replay_offsets.txt")
    args = ap.parse_args()

    cfg = args.config
    roadnet = args.roadnet
    intersections = ["intersection_1_1", "intersection_2_1", "intersection_3_1"]

    phase_times: Dict[str, List[float]] = {
        iid: _load_phase_times(roadnet, iid) for iid in intersections
    }
    offsets: Dict[str, float] = {
        "intersection_1_1": float(args.offset_1_1),
        "intersection_2_1": float(args.offset_2_1),
        "intersection_3_1": float(args.offset_3_1),
    }

    eng = cityflow.Engine(cfg, thread_num=int(args.thread_num))
    # saveReplay must be true in config; replay_out must be relative so dir+path is valid.
    eng.set_save_replay(True)
    eng.set_replay_file(args.replay_out)

    for _ in range(int(args.steps)):
        t = float(eng.get_current_time())
        for iid in intersections:
            ph = _phase_at_time(phase_times[iid], t + offsets[iid])
            eng.set_tl_phase(iid, int(ph))
        eng.next_step()

    # Let Engine destructor flush replay when process exits.


if __name__ == "__main__":
    # Run from repo root so config relative paths resolve.
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main()

