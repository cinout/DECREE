fp = open("resultfinal_cliptext_20251019_124017.txt", "r")
lines = fp.readlines()

# positive (1): backdoor
# negative (0): clean

tp, tn, fp, fn = 0, 0, 0, 0
tp_list, tn_list, fp_list, fn_list = [], [], [], []
total_clean, total_backdoor = 0, 0

threshold = 0.1

for line in lines:
    contents = line.split(",")
    id, gt, pl1_norm = contents[0], contents[1], float(contents[3])

    pred = 1 if pl1_norm < threshold else 0
    gt = 1 if gt == "backdoor" else 0

    if gt == 0:
        # gt: 0
        total_clean += 1
        if pred == 0:
            tn += 1
            tn_list.append(id)
        else:
            fp += 1
            fp_list.append(id)
    else:
        # gt: 1
        total_backdoor += 1
        if pred == 0:
            fn += 1
            fn_list.append(id)
        else:
            tp += 1
            tp_list.append(id)

acc = (tp + tn) / (tp + tn + fp + fn)

print(f"TP\tFP\tFN\tTN\tACC\n")
print(f"{tp}\t{fp}\t{fn}\t{tn}\t{acc*100:.1f}")
print("--------------")
print(f"Total Clean: {total_clean}, Total Backdoor: {total_backdoor}")
print("--------------")
print(f"TP: {tp_list}")
print("--------------")
print(f"FP: {fp_list}")
print("--------------")
print(f"FN: {fn_list}")
print("--------------")
print(f"TN: {tn_list}")
print("--------------")
