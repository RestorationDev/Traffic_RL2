import cityflow

# See mydata_1x3/run_data_1x3.py — set_replay_file("/app/...") breaks logging (dir + path).
engine = cityflow.Engine("/app/mydata/config.json", thread_num=1)

print("Starting simulation")
for step in range(10000):
    if step % 1000 == 0:
        print(f"Step: {step}")
    engine.next_step()
print("Simulation done")