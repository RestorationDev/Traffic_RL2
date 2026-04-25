import cityflow

# Replay paths come from config.json (saveReplay, replayLogFile).
# Do not call set_replay_file with an absolute path: CityFlow prepends
# config "dir" (e.g. "./"), producing ".//app/..." and the log stream fails to open.
#
# For fixed-time signals: rlTrafficLight must be false in config.json so the engine
# advances lightphases via passTime(). If it is true, phases never change unless you
# call set_tl_phase() each step (RL). Use config_rl.json when training with train.py.
#
# Replay viewer: load replay_roadnet.json (roadnetLogFile), NOT roadnet_1X3.json —
# see VIEW_REPLAY.txt in this folder.
engine = cityflow.Engine("/app/mydata_1x3/config.json", thread_num=1)

print("Starting simulation")
for step in range(10000):
    if step % 1000 == 0:
        print(f"Step: {step}")
    engine.next_step()
print("Simulation done")