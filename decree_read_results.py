from sklearn.metrics import roc_auc_score

# positive (1): backdoor
# negative (0): clean

file_names = [
    "results/results_openclip_q0.3-0.7_wl2_0.txt",
    "results/results_openclip_q0.3-0.7_wl2_separate.txt",
]

triggers = ["badnets", "wanet", "nashville", "blend", "sig"]

for file_name in file_names:
    print(f"============={file_name}=============")
    file_handler = open(file_name, "r")
    lines = file_handler.readlines()

    tp, tn, fp, fn = 0, 0, 0, 0
    tp_list, tn_list, fp_list, fn_list = [], [], [], []
    total_clean, total_backdoor = 0, 0
    y_true = []
    y_mask_norm_neg = []
    y_l2_norm_dist_quantile_normalized = []

    ids = []
    trigger_dict = dict()

    # threshold = 0.1
    threshold = 20.0

    for line in lines:
        contents = line.split(",")
        id, gt, pl1_norm, l2_norm_quantile = (
            contents[0],
            int(contents[1]),
            float(contents[2]),
            float(contents[3]),
        )

        # print(l2_norm_quantile)

        pred = 1 if l2_norm_quantile > threshold else 0

        ids.append(id)
        y_true.append(gt)
        y_mask_norm_neg.append(-pl1_norm)
        y_l2_norm_dist_quantile_normalized.append(l2_norm_quantile)

        if gt == 0:
            # clean encoder
            total_clean += 1
            if pred == 0:
                tn += 1
                tn_list.append(id)
            else:
                fp += 1
                fp_list.append(id)
        else:
            # poisoned encoder
            total_backdoor += 1
            if pred == 0:
                fn += 1
                fn_list.append(id)
            else:
                tp += 1
                tp_list.append(id)

    acc = (tp + tn) / (tp + tn + fp + fn)
    auc_mask_norm_neg = roc_auc_score(y_true, y_mask_norm_neg)
    auc_l2_norm_dist_quantile_normalized = roc_auc_score(
        y_true, y_l2_norm_dist_quantile_normalized
    )

    print("-------OVERALL-------")
    print(f"AUROC(%) Our Method: {auc_l2_norm_dist_quantile_normalized*100:.1f}")
    print(f"AUROC(%) DECREE: {auc_mask_norm_neg*100:.1f}")

    # scores by categories
    vit_encoder_indices = [i for (i, id) in enumerate(ids) if "vit" in id.lower()]
    resnet_encoder_indices = [
        i for (i, id) in enumerate(ids) if "vit" not in id.lower()
    ]
    y_true_vit = [y_true[i] for i in vit_encoder_indices]
    y_true_resnet = [y_true[i] for i in resnet_encoder_indices]

    # triggered encoders
    for trigger in triggers:
        vit_encoder_with_trigger_indices = [
            i
            for (i, id) in enumerate(ids)
            if "vit" in id.lower() and trigger in id.lower()
        ]
        resnet_encoder_with_trigger_indices = [
            i
            for (i, id) in enumerate(ids)
            if "rn" in id.lower() and trigger in id.lower()
        ]
        trigger_dict[trigger] = {
            "vit": vit_encoder_with_trigger_indices,
            "rn": resnet_encoder_with_trigger_indices,
        }
    # clean encoder
    clean_vit_encoder_indices = [
        i
        for (i, (gt, id)) in enumerate(zip(y_true, ids))
        if gt == 0 and "vit" in id.lower()
    ]
    clean_resnet_encoder_indices = [
        i
        for (i, (gt, id)) in enumerate(zip(y_true, ids))
        if gt == 0 and "rn" in id.lower()
    ]

    print("-------Our Method-------")
    y_l2_norm_dist_quantile_normalized_vit = [
        y_l2_norm_dist_quantile_normalized[i] for i in vit_encoder_indices
    ]
    y_l2_norm_dist_quantile_normalized_resnet = [
        y_l2_norm_dist_quantile_normalized[i] for i in resnet_encoder_indices
    ]
    auc_vit = roc_auc_score(y_true_vit, y_l2_norm_dist_quantile_normalized_vit)
    auc_resnet = roc_auc_score(y_true_resnet, y_l2_norm_dist_quantile_normalized_resnet)
    print(f"auc_vit (ALL): {auc_vit*100:.1f}")
    print(f"auc_resnet (ALL): {auc_resnet*100:.1f}")
    print("-----")

    # trigger-level
    clean_vit_encoder_scores = [
        y_l2_norm_dist_quantile_normalized[i] for i in clean_vit_encoder_indices
    ]
    clean_resnet_encoder_scores = [
        y_l2_norm_dist_quantile_normalized[i] for i in clean_resnet_encoder_indices
    ]
    for trigger in triggers:
        vit_encoder_with_trigger_indices = trigger_dict[trigger]["vit"]
        resnet_encoder_with_trigger_indices = trigger_dict[trigger]["rn"]
        vit_encoder_with_trigger_scores = [
            y_l2_norm_dist_quantile_normalized[i]
            for i in vit_encoder_with_trigger_indices
        ]
        resnet_encoder_with_trigger_scores = [
            y_l2_norm_dist_quantile_normalized[i]
            for i in resnet_encoder_with_trigger_indices
        ]
        auc_vit = roc_auc_score(
            [y_true[i] for i in clean_vit_encoder_indices]
            + [y_true[i] for i in vit_encoder_with_trigger_indices],
            clean_vit_encoder_scores + vit_encoder_with_trigger_scores,
        )
        auc_resnet = roc_auc_score(
            [y_true[i] for i in clean_resnet_encoder_indices]
            + [y_true[i] for i in resnet_encoder_with_trigger_indices],
            clean_resnet_encoder_scores + resnet_encoder_with_trigger_scores,
        )

        print(f"auc_vit ({trigger}): {auc_vit*100:.1f}")
        print(f"auc_resnet ({trigger}): {auc_resnet*100:.1f}")
        print("-----")

    print("-------DECREE-------")
    y_mask_norm_neg_vit = [y_mask_norm_neg[i] for i in vit_encoder_indices]
    y_mask_norm_neg_resnet = [y_mask_norm_neg[i] for i in resnet_encoder_indices]
    auc_vit = roc_auc_score(y_true_vit, y_mask_norm_neg_vit)
    auc_resnet = roc_auc_score(y_true_resnet, y_mask_norm_neg_resnet)
    print(f"auc_vit (ALL): {auc_vit*100:.1f}")
    print(f"auc_resnet (ALL): {auc_resnet*100:.1f}")
    print("-----")

    # trigger-level
    clean_vit_encoder_scores = [y_mask_norm_neg[i] for i in clean_vit_encoder_indices]
    clean_resnet_encoder_scores = [
        y_mask_norm_neg[i] for i in clean_resnet_encoder_indices
    ]
    for trigger in triggers:
        vit_encoder_with_trigger_indices = trigger_dict[trigger]["vit"]
        resnet_encoder_with_trigger_indices = trigger_dict[trigger]["rn"]
        vit_encoder_with_trigger_scores = [
            y_mask_norm_neg[i] for i in vit_encoder_with_trigger_indices
        ]
        resnet_encoder_with_trigger_scores = [
            y_mask_norm_neg[i] for i in resnet_encoder_with_trigger_indices
        ]
        auc_vit = roc_auc_score(
            [y_true[i] for i in clean_vit_encoder_indices]
            + [y_true[i] for i in vit_encoder_with_trigger_indices],
            clean_vit_encoder_scores + vit_encoder_with_trigger_scores,
        )
        auc_resnet = roc_auc_score(
            [y_true[i] for i in clean_resnet_encoder_indices]
            + [y_true[i] for i in resnet_encoder_with_trigger_indices],
            clean_resnet_encoder_scores + resnet_encoder_with_trigger_scores,
        )

        print(f"auc_vit ({trigger}): {auc_vit*100:.1f}")
        print(f"auc_resnet ({trigger}): {auc_resnet*100:.1f}")
        print("-----")

    file_handler.close()
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
