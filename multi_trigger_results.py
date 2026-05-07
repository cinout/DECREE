"""
ViT-B-16_laion400m_e32
RN101_yfcc15m
ViT-B-32_laion400m_e31
ViT-B-32_laion2b_e16
RN101_openai
ViT-B-16_laion2b_s34b_b88k
ViT-B-16_laion400m_e31
ViT-B-32_openai
ViT-B-16_openai
ViT-B-32_laion400m_e32
"""

from sklearn.metrics import roc_auc_score, average_precision_score

gt = [0] * 10 + [1] * 10

# clean

ours_clean = [
    0.3022,
    0.1975,
    0.3456,
    0.3942,
    0.6262,
    0.3142,
    0.3113,
    0.5238,
    0.5550,
    0.3456,
]

decree_clean = [
    0.0591,
    0.0217,
    0.0505,
    0.0810,
    0.3242,
    0.0107,
    0.0595,
    0.0389,
    0.0168,
    0.0359,
]

masa_clean = [
    78.8944,
    228.3780,
    130.2815,
    117.4729,
    254.2496,
    125.6376,
    85.2105,
    196.4009,
    86.2858,
    110.4866,
]

# multi-trgger, single target
ours_bd_single_target = [
    0.6383,
    0.7930,
    0.7168,
    0.7762,
    0.7287,
    0.7571,
    0.7746,
    0.7009,
    0.5518,
    0.8124,
]

decree_single_target = [
    0.0207,
    0.0417,
    0.1070,
    0.0060,
    0.0128,
    0.0387,
    0.0215,
    0.0235,
    0.0324,
    0.0160,
]

masa_single_target = [
    122.6615,
    121.9619,
    250.8858,
    185.1713,
    157.6792,
    167.2528,
    173.0908,
    79.9427,
    56.2363,
    167.9464,
]


# multi-trgger, multi target
ours_bd_multi_target = [
    0.7902,
    0.7680,
    0.6523,
    0.7693,
    0.7390,
    0.7740,
    0.8113,
    0.5303,
    0.6678,
    0.7260,
]

decree_multi_target = [
    0.0254,
    0.0189,
    0.0272,
    0.0166,
    0.0581,
    0.0162,
    0.0500,
    0.0677,
    0.0629,
    0.0270,
]
masa_multi_target = [
    86.5556,
    153.0778,
    139.5970,
    87.3229,
    246.5901,
    73.3429,
    177.8462,
    205.8846,
    96.9094,
    129.5333,
]


auc_ours_single_target = roc_auc_score(gt, ours_clean + ours_bd_single_target)
ap_ours_single_target = average_precision_score(gt, ours_clean + ours_bd_single_target)

auc_decree_single_target = roc_auc_score(gt, decree_clean + decree_single_target)
ap_decree_single_target = average_precision_score(
    gt, decree_clean + decree_single_target
)

auc_masa_single_target = roc_auc_score(gt, masa_clean + masa_single_target)
ap_masa_single_target = average_precision_score(gt, masa_clean + masa_single_target)


eminspector_single_target_pred = [
    -665,
    -737,
    -785,
    785,
    785,
    -751,
    -737,
    785,
    785,
    -591,
    -785,
    385,
    -629,
    -781,
    -571,
    785,
    -697,
    785,
    -727,
    165,
]
eminspector_multi_target_pred = [
    -659,
    -739,
    -785,
    785,
    785,
    -745,
    -725,
    785,
    785,
    -627,
    233,
    -695,
    785,
    -549,
    -647,
    -767,
    275,
    -691,
    -785,
    785,
]
auc_eminspector_single_target = roc_auc_score(gt, eminspector_single_target_pred)
ap_eminspector_single_target = average_precision_score(
    gt, eminspector_single_target_pred
)

auc_ours_multi_target = roc_auc_score(gt, ours_clean + ours_bd_multi_target)
ap_ours_multi_target = average_precision_score(gt, ours_clean + ours_bd_multi_target)

auc_decree_multi_target = roc_auc_score(gt, decree_clean + decree_multi_target)
ap_decree_multi_target = average_precision_score(gt, decree_clean + decree_multi_target)

auc_masa_multi_target = roc_auc_score(gt, masa_clean + masa_multi_target)
ap_masa_multi_target = average_precision_score(gt, masa_clean + masa_multi_target)

auc_eminspector_multi_target = roc_auc_score(gt, eminspector_multi_target_pred)
ap_eminspector_multi_target = average_precision_score(gt, eminspector_multi_target_pred)

print("-------SINGLE TARGET-------")
print(
    f"AUROC(%)/AP(%) Our Method: {auc_ours_single_target*100:.1f}/{ap_ours_single_target*100:.1f}"
)
print(
    f"AUROC(%)/AP(%) DECREE: {auc_decree_single_target*100:.1f}/{ap_decree_single_target*100:.1f}"
)
print(
    f"AUROC(%)/AP(%) MASA: {auc_masa_single_target*100:.1f}/{ap_masa_single_target*100:.1f}"
)
print(
    f"AUROC(%)/AP(%) EMINspector: {auc_eminspector_single_target*100:.1f}/{ap_eminspector_single_target*100:.1f}"
)
print("-------MULTI TARGET-------")
print(
    f"AUROC(%)/AP(%) Our Method: {auc_ours_multi_target*100:.1f}/{ap_ours_multi_target*100:.1f}"
)
print(
    f"AUROC(%)/AP(%) DECREE: {auc_decree_multi_target*100:.1f}/{ap_decree_multi_target*100:.1f}"
)
print(
    f"AUROC(%)/AP(%) MASA: {auc_masa_multi_target*100:.1f}/{ap_masa_multi_target*100:.1f}"
)
print(
    f"AUROC(%)/AP(%) EMINspector: {auc_eminspector_multi_target*100:.1f}/{ap_eminspector_multi_target*100:.1f}"
)
