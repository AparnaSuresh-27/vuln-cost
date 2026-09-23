import json

counts = []
with open("data/final_benchmark.jsonl") as f:
    for line in f:
        r = json.loads(line)
        for prefix in ("vulnerable", "non_vulnerable"):
            bodies = r.get(f"{prefix}_function_bodies") or {}
            target_names = set((r.get(f"{prefix}_caller_graph") or {}).keys()) | \
                           set((r.get(f"{prefix}_callee_graph") or {}).keys())
            neighbour_bodies = [n for n in bodies if n not in target_names]
            counts.append(len(neighbour_bodies))

print(f"records: {len(counts)}")
print(f"with neighbour bodies: {sum(1 for c in counts if c > 0)}")
print(f"max neighbours: {max(counts)}, mean: {sum(counts)/len(counts):.1f}")

