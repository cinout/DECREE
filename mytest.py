file_handler = open("results/results_openclip_cossim_comprehensive.txt", "r")
lines = file_handler.readlines()
output_file = "z_results_detection_scores.txt"
output_acc_asr_file_handle = open(output_file, "w", encoding="utf-8")

for line in lines:
    contents = line.split(",")
    id, score = (
        contents[0],
        float(contents[3]),
    )
    output_acc_asr_file_handle.write(f"{id}\t{score*100:.2f}\n")

file_handler.close()
