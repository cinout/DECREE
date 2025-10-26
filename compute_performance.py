from sklearn.metrics import roc_auc_score

fp = open("resultfinal_cliptext_distance.txt", "r")
lines = fp.readlines()

# positive (1): backdoor
# negative (0): clean

tp, tn, fp, fn = 0, 0, 0, 0
tp_list, tn_list, fp_list, fn_list = [], [], [], []
total_clean, total_backdoor = 0, 0
y_true = []
y_mask_norm_neg = []
y_l2_norm_dist = []
y_cos_sim_neg = []

threshold = 0.1

for line in lines:
    contents = line.split(",")
    id, gt, pl1_norm, l2_norm, cos_sim = (
        contents[0],
        contents[1],
        float(contents[3]),
        float(contents[6]),
        float(contents[7]),
    )

    print(cos_sim)

    pred = 1 if pl1_norm < threshold else 0
    gt = 1 if gt == "backdoor" else 0

    y_true.append(gt)
    y_mask_norm_neg.append(-pl1_norm)
    y_l2_norm_dist.append(l2_norm)
    y_cos_sim_neg.append(-cos_sim)

    if gt == 0:

        total_clean += 1
        if pred == 0:
            tn += 1
            tn_list.append(id)
        else:
            fp += 1
            fp_list.append(id)
    else:

        total_backdoor += 1
        if pred == 0:
            fn += 1
            fn_list.append(id)
        else:
            tp += 1
            tp_list.append(id)

acc = (tp + tn) / (tp + tn + fp + fn)
auc_mask_norm_neg = roc_auc_score(y_true, y_mask_norm_neg)
auc_l2_norm_dist = roc_auc_score(y_true, y_l2_norm_dist)
auc_cos_sim_neg = roc_auc_score(y_true, y_cos_sim_neg)

# print("--------------")
# print(f"AUROC(%) Mask Norm Neg: {auc_mask_norm_neg*100:.1f}")
# print(f"AUROC(%) L2-Norm Dist: {auc_l2_norm_dist*100:.1f}")
# print(f"AUROC(%) Cosine Sim Neg: {auc_cos_sim_neg*100:.1f}")

# print(f"TP\tFP\tFN\tTN\tACC\n")
# print(f"{tp}\t{fp}\t{fn}\t{tn}\t{acc*100:.1f}")
# print("--------------")
# print(f"Total Clean: {total_clean}, Total Backdoor: {total_backdoor}")
# print("--------------")
# print(f"TP: {tp_list}")
# print("--------------")
# print(f"FP: {fp_list}")
# print("--------------")
# print(f"FN: {fn_list}")
# print("--------------")
# print(f"TN: {tn_list}")
# print("--------------")
