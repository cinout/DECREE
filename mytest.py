sty = "OPENCLIP_ViT-B-16_laion400m_e32_triggers_nashville_ftrojan_wanet_targets_dhole_paintbrush_goblet_trainsetp_0.1_epoch_0.pth"

name_split = sty.split("_")
arch = name_split[1]
key = "_".join(name_split[2:]).split("_trigger")[0]

print(arch)
print(key)
