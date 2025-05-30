import json
import os

def  get_incoming_lanes_by_intersection(roadnet_path):
    with open(roadnet_path, "r") as f:
        data = json.load(f)
    
    intersections = data["intersections"]
    roads = data["roads"]

    incoming_lanes = {}

    for inter in intersections:
        if not inter["virtual"]:
            incoming_lanes[inter["id"]] = []

    for road in roads:
        end_inter = road["endIntersection"]
        if end_inter in incoming_lanes:
            for i, lane in enumerate(road["lanes"]):
                lane_id = f"{road['id']}_{i}"
                incoming_lanes[end_inter].append(lane_id)
    return incoming_lanes


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # path to envs/
    roadnet_path = os.path.join(base_dir, "..", "mydata", "roadnet.json")
    incoming = get_incoming_lanes_by_intersection(roadnet_path)
    print(json.dumps(incoming, indent=2))
    with open("incoming_lanes.json", "w") as f:
        json.dump(incoming, f, indent=2)

    print("Saved to incoming_lanes.json")