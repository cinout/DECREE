from sklearn.metrics import roc_auc_score

# positive (1): backdoor
# negative (0): clean

file_names = [
    "results/*concoct2.txt",
    # "results/results_invert_onlycosine_no_mask.txt",
]


for file_name in file_names:
    print(f"============={file_name}=============")
    file_handler = open(file_name, "r")
    lines = file_handler.readlines()

    y_true = []
    y_mask_norm_neg = []
    y_l2_norm_dist_quantile_normalized = []

    ids = []

    for line in lines:
        contents = line.split(",")
        id, gt, pl1_norm, l2_norm_quantile = (
            contents[0],
            int(contents[1]),
            float(contents[2]),
            float(contents[3]),
        )

        # print(l2_norm_quantile)

        ids.append(id)
        y_true.append(gt)
        y_mask_norm_neg.append(-pl1_norm)
        y_l2_norm_dist_quantile_normalized.append(l2_norm_quantile)

    auc_mask_norm_neg = roc_auc_score(y_true, y_mask_norm_neg)
    auc_l2_norm_dist_quantile_normalized = roc_auc_score(
        y_true, y_l2_norm_dist_quantile_normalized
    )

    print("-------OVERALL-------")
    print(f"AUROC(%) Our Method: {auc_l2_norm_dist_quantile_normalized*100:.1f}")
    print(f"AUROC(%) DECREE: {auc_mask_norm_neg*100:.1f}")

    # # scores by categories
    # vit_encoder_indices = [i for (i, id) in enumerate(ids) if "vit" in id.lower()]
    # resnet_encoder_indices = [
    #     i for (i, id) in enumerate(ids) if "vit" not in id.lower()
    # ]
    # y_true_vit = [y_true[i] for i in vit_encoder_indices]
    # y_true_resnet = [y_true[i] for i in resnet_encoder_indices]

    # # clean encoder
    # clean_vit_encoder_indices = [
    #     i
    #     for (i, (gt, id)) in enumerate(zip(y_true, ids))
    #     if gt == 0 and "vit" in id.lower()
    # ]
    # clean_resnet_encoder_indices = [
    #     i
    #     for (i, (gt, id)) in enumerate(zip(y_true, ids))
    #     if gt == 0 and "rn" in id.lower()
    # ]

    # print("-------Our Method-------")
    # y_l2_norm_dist_quantile_normalized_vit = [
    #     y_l2_norm_dist_quantile_normalized[i] for i in vit_encoder_indices
    # ]
    # y_l2_norm_dist_quantile_normalized_resnet = [
    #     y_l2_norm_dist_quantile_normalized[i] for i in resnet_encoder_indices
    # ]
    # auc_vit = roc_auc_score(y_true_vit, y_l2_norm_dist_quantile_normalized_vit)
    # auc_resnet = roc_auc_score(y_true_resnet, y_l2_norm_dist_quantile_normalized_resnet)
    # print(f"auc_vit (ALL): {auc_vit*100:.1f}")
    # print(f"auc_resnet (ALL): {auc_resnet*100:.1f}")
    # print("-----")

    # # trigger-level
    # clean_vit_encoder_scores = [
    #     y_l2_norm_dist_quantile_normalized[i] for i in clean_vit_encoder_indices
    # ]
    # clean_resnet_encoder_scores = [
    #     y_l2_norm_dist_quantile_normalized[i] for i in clean_resnet_encoder_indices
    # ]

    # print("-------DECREE-------")
    # y_mask_norm_neg_vit = [y_mask_norm_neg[i] for i in vit_encoder_indices]
    # y_mask_norm_neg_resnet = [y_mask_norm_neg[i] for i in resnet_encoder_indices]
    # auc_vit = roc_auc_score(y_true_vit, y_mask_norm_neg_vit)
    # auc_resnet = roc_auc_score(y_true_resnet, y_mask_norm_neg_resnet)
    # print(f"auc_vit (ALL): {auc_vit*100:.1f}")
    # print(f"auc_resnet (ALL): {auc_resnet*100:.1f}")
    # print("-----")

    file_handler.close()
