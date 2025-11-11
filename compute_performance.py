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
y_l2_norm_dist_quantile_normalized = []
indices_of_vit = []
ids = []

# threshold = 0.1
threshold = 20.0

for line in lines:
    contents = line.split(",")
    id, gt, pl1_norm, l2_norm, cos_sim, l2_norm_quantile = (
        contents[0],
        contents[1],
        float(contents[3]),
        float(contents[6]),
        float(contents[7]),
        float(contents[8]),
    )

    # print(l2_norm_quantile)

    pred = 1 if l2_norm_quantile > threshold else 0
    gt = 1 if gt == "backdoor" else 0
    ids.append(id)

    y_true.append(gt)
    y_mask_norm_neg.append(-pl1_norm)
    y_l2_norm_dist.append(l2_norm)
    y_cos_sim_neg.append(-cos_sim)
    y_l2_norm_dist_quantile_normalized.append(l2_norm_quantile)

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


auc_l2_norm_dist_quantile_normalized = roc_auc_score(
    y_true, y_l2_norm_dist_quantile_normalized
)


print("--------------")
print(f"AUROC(%) Mask Norm Neg: {auc_mask_norm_neg*100:.1f}")
print(f"AUROC(%) L2-Norm Dist: {auc_l2_norm_dist*100:.1f}")
print(f"AUROC(%) Cosine Sim Neg: {auc_cos_sim_neg*100:.1f}")
print(
    f"AUROC(%) L2-Norm Dist (Quantile Norm): {auc_l2_norm_dist_quantile_normalized*100:.1f}"
)

vit_encoder_indices = [i for (i, id) in enumerate(ids) if "vit" in id.lower()]
resnet_encoder_indices = [i for (i, id) in enumerate(ids) if "vit" not in id.lower()]

y_true_vit = [y_true[i] for i in vit_encoder_indices]
y_true_resnet = [y_true[i] for i in resnet_encoder_indices]


print("--------------")
print(f"vit count: {len(y_true_vit)}")
print(f"resnet count: {len(y_true_resnet)}")
print("-------Our Method-------")
y_l2_norm_dist_quantile_normalized_vit = [
    y_l2_norm_dist_quantile_normalized[i] for i in vit_encoder_indices
]
y_l2_norm_dist_quantile_normalized_resnet = [
    y_l2_norm_dist_quantile_normalized[i] for i in resnet_encoder_indices
]
auc_vit = roc_auc_score(y_true_vit, y_l2_norm_dist_quantile_normalized_vit)
auc_resnet = roc_auc_score(y_true_resnet, y_l2_norm_dist_quantile_normalized_resnet)
print(f"auc_vit: {auc_vit*100:.1f}")
print(f"auc_resnet: {auc_resnet*100:.1f}")

print("-------DECREE-------")
y_mask_norm_neg_vit = [y_mask_norm_neg[i] for i in vit_encoder_indices]
y_mask_norm_neg_resnet = [y_mask_norm_neg[i] for i in resnet_encoder_indices]
auc_vit = roc_auc_score(y_true_vit, y_mask_norm_neg_vit)
auc_resnet = roc_auc_score(y_true_resnet, y_mask_norm_neg_resnet)
print(f"auc_vit: {auc_vit*100:.1f}")
print(f"auc_resnet: {auc_resnet*100:.1f}")

# print(f"TP\tFP\tFN\tTN\tACC(%)\n")
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
