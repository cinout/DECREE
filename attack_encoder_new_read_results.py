"""
Read results from eval_attack.py
"""

import re


file_path = "slurm-19584075-[trigger_wanet].out"
output_file = "z_results_attack_encoder.txt"
output_acc_asr_file_handle = open(output_file, "w", encoding="utf-8")

metric_pattern = r".*Clean Acc Top-1:\s*([\d.]+)\s*ASR Top-1:\s*([\d.]+).*"
id_pattern = r">>> Evaluate encoder \S+ (\S+)$"

benchmark_pattern = (
    r"\[After Epoch 0\].*Clean ACC:\s*([\d.]+);\s*Backdoor ASR:\s*([\d.]+)"
)

metric_store = []
# id_store = []
with open(file_path, "r") as f:
    for line in f:

        # benchmark
        benchmark_pattern_match = re.search(benchmark_pattern, line)
        if benchmark_pattern_match:
            acc = float(benchmark_pattern_match.group(1)) * 100
            asr = float(benchmark_pattern_match.group(2)) * 100

            metric_store.append((acc, asr))

        # # id
        # id_pattern_match = re.search(id_pattern, line)
        # if id_pattern_match:
        #     id = id_pattern_match.group(1)
        #     id_store.append(id)

output_lines = [f"{metric[0]:.2f}\t{metric[1]:.2f}\n" for metric in metric_store]
# output_lines = [
#     f"{id}\t{metric[0]:.2f}\t{metric[1]:.2f}\n"
#     for id, metric in zip(id_store, metric_store)
# ]

for line in output_lines:
    output_acc_asr_file_handle.write(line)
