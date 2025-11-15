import re


file_path = "slurm-18604317-[PART_2].out"
output_file = "z_results.txt"
output_acc_asr_file_handle = open(output_file, "w", encoding="utf-8")

metric_pattern = r".*Clean Acc Top-1:\s*([\d.]+)\s*ASR Top-1:\s*([\d.]+).*"
id_pattern = r">>> Evaluate encoder \S+ (\S+)$"

metric_store = []
id_store = []
with open(file_path, "r") as f:
    for line in f:

        # metric
        metric_pattern_match = re.search(metric_pattern, line)
        if metric_pattern_match:
            acc = float(metric_pattern_match.group(1)) * 100
            asr = float(metric_pattern_match.group(2)) * 100

            metric_store.append((acc, asr))

        # id
        id_pattern_match = re.search(id_pattern, line)
        if id_pattern_match:
            id = id_pattern_match.group(1)
            id_store.append(id)

output_lines = [
    f"{id}\t{metric[0]:.2f}\t{metric[1]:.2f}\n"
    for id, metric in zip(id_store, metric_store)
]

for line in output_lines:
    output_acc_asr_file_handle.write(line)
