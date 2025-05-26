import cityflow

engine = cityflow.Engine("/app/mydata/config.json", thread_num=1)

engine.set_save_replay(True)
engine.set_replay_file("/app/mydata/replay.txt")

print("Starting simulation")
for step in range(10000):
    if step % 1000 == 0:
        print(f"Step: {step}")
    engine.next_step()
print("Simulation done")