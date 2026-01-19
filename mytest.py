"""
read explain results
"""

file_handler = open("explain_result.txt", "r")
lines = file_handler.readlines()
output_file = "z_explain_results.txt"
output_acc_asr_file_handle = open(output_file, "w", encoding="utf-8")

for line in lines:
    contents = line.split(",")
    id, bd_simulated_with_clean_avg, bd_simulated_pairwise_avg = (
        contents[0],
        float(contents[2]),
        float(contents[3]),
    )
    if len(contents) > 4:
        bd_real_with_clean_avg, bd_real_pairwise_avg, bd_real_with_bd_simulated_avg = (
            float(contents[4]),
            float(contents[5]),
            float(contents[6]),
        )
        output_acc_asr_file_handle.write(
            f"{id}\t{bd_simulated_with_clean_avg:.2f}\t{bd_simulated_pairwise_avg:.2f}\t{bd_real_with_clean_avg:.2f}\t{bd_real_pairwise_avg:.2f}\t{bd_real_with_bd_simulated_avg:.2f}\n"
        )
    else:

        output_acc_asr_file_handle.write(
            f"{id}\t{bd_simulated_with_clean_avg:.2f}\t{bd_simulated_pairwise_avg:.2f}\n"
        )

file_handler.close()

"""
read decree's results
"""
# file_handler = open("results/*results_oldset_cossim.txt", "r")
# lines = file_handler.readlines()
# output_file = "z_results_detection_scores.txt"
# output_acc_asr_file_handle = open(output_file, "w", encoding="utf-8")

# for line in lines:
#     contents = line.split(",")
#     id, score = (
#         contents[0],
#         float(contents[3]),
#     )
#     output_acc_asr_file_handle.write(f"{id}\t{score:.2f}\n")

# file_handler.close()
