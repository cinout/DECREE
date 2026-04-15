import matplotlib.pyplot as plt
import numpy as np
import torch


# cos_sim_all = torch.load("cos_sim_all_20260412_232310_22_47.pt", weights_only=True)

# what_i_need = [
#     "OPENCLIP_RN50_yfcc15m",
#     "OPENCLIP_BD_badnets_trainsetp_0.01_epoch_0_RN50_yfcc15m",
#     #
#     "OPENCLIP_RN101_openai",
#     "OPENCLIP_BD_wanet_trainsetp_0.05_epoch_0_RN101_openai",
#     #
#     "OPENCLIP_ViT-B-16_laion400m_e31",
#     "OPENCLIP_BD_nashville_trainsetp_0.05_epoch_0_ViT-B-16_laion400m_e31",
#     #
#     "OPENCLIP_ViT-B-32_laion2b_e16",
#     "OPENCLIP_BD_ftrojan_trainsetp_0.05_epoch_0_ViT-B-32_laion2b_e16",
#     #
#     "OPENCLIP_RN50x4_openai",
#     "OPENCLIP_BD_blend_trainsetp_0.2_epoch_0_RN50x4_openai",
#     #
#     "OPENCLIP_ViT-L-14_metaclip_400m",
#     "OPENCLIP_BD_sig_trainsetp_0.05_epoch_0_ViT-L-14_metaclip_400m",
# ]

# for need in what_i_need:
#     print(f"{need}:\n")
#     print(cos_sim_all[need][:4])

data = {
    "RN-50 (Badnets)": {
        "clean": [
            0.3140680491924286,
            0.12553638219833374,
            0.30457326769828796,
            0.21961840987205505,
        ],
        "bd": [
            0.6790051460266113,
            0.5412316918373108,
            0.6121404767036438,
            0.6401400566101074,
        ],
    },
    "RN-101 (WaNet)": {
        "clean": [
            0.5729129314422607,
            0.5292673110961914,
            0.58307945728302,
            0.610237181186676,
        ],
        "bd": [
            0.7311366200447083,
            0.6884193420410156,
            0.7075178623199463,
            0.7349272966384888,
        ],
    },
    "ViT-B-16 (Nashville)": {
        "clean": [
            0.2843892276287079,
            0.23171775043010712,
            0.23819474875926971,
            0.33931174874305725,
        ],
        "bd": [
            0.7905641794204712,
            0.7624521255493164,
            0.7903865575790405,
            0.8256130814552307,
        ],
    },
    "ViT-B-32 (FTrojan)": {
        "clean": [
            0.2885624170303345,
            0.34397953748703003,
            0.2845882773399353,
            0.4163075387477875,
        ],
        "bd": [
            0.7046461701393127,
            0.7671917676925659,
            0.7317706346511841,
            0.7625575661659241,
        ],
    },
    "RN50x4 (Blend)": {
        "clean": [
            0.5206984877586365,
            0.4404650330543518,
            0.5558688640594482,
            0.5305046439170837,
        ],
        "bd": [
            0.6893668174743652,
            0.7050595283508301,
            0.6941959857940674,
            0.6606220006942749,
        ],
    },
    "ViT-L-14 (SIG)": {
        "clean": [
            0.46729952096939087,
            0.4949718713760376,
            0.45178115367889404,
            0.5381107330322266,
        ],
        "bd": [
            0.7301150560379028,
            0.7186892628669739,
            0.6560578346252441,
            0.7539721727371216,
        ],
    },
}


def plot_grouped_pairs(data, title=None, xlim=(0, 1), y_spacing=0.6):
    """Plot groups on the y-axis and cosine similarity (0..1) on the x-axis.

    Each group has three pairs of (clean, bd) values. Different marker shapes
    and colors identify the three pairs. Clean markers are hollow; bd markers
    are filled. A thin line connects each pair.
    """
    groups = list(data.keys())
    # Ensure order: top to bottom should be group_A ... group_D if present
    # If the user provides a different order, keep the dict order but we want
    # top-to-bottom as given in the dict.

    n_groups = len(groups)
    # Scale y positions by y_spacing to reduce vertical gaps between groups
    y_positions = (np.arange(n_groups - 1, -1, -1) * y_spacing).tolist()

    fig_height = max(2, 1 + n_groups * y_spacing)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    marker_shapes = ["o", "s", "^", "D"]
    colors = ["C0", "C1", "C2", "C3"]
    # Offset markers slightly; scale by y_spacing so offsets remain proportional
    pair_offsets = np.linspace(-0.12, 0.12, len(marker_shapes)) * y_spacing

    # We'll build custom legend handles: first row = hollow markers + 'clean encoder'
    # second row = filled markers + 'backdoored encoder'
    legend_handles = []
    for gi, group in enumerate(groups):
        y = y_positions[gi]
        clean_vals = data[group]["clean"]
        bd_vals = data[group]["bd"]
        for i, (c_val, b_val) in enumerate(zip(clean_vals, bd_vals)):
            mx = marker_shapes[i % len(marker_shapes)]
            col = colors[i % len(colors)]
            y_off = pair_offsets[i % len(pair_offsets)]

            # Plot clean (hollow)
            ax.plot(
                c_val,
                y + y_off,
                marker=mx,
                markersize=8,
                markeredgecolor=col,
                markerfacecolor="none",
                linestyle="",
                zorder=3,
            )

            # Plot bd (filled)
            ax.plot(
                b_val,
                y + y_off,
                marker=mx,
                markersize=8,
                markeredgecolor=col,
                markerfacecolor=col,
                linestyle="",
                zorder=4,
            )

            # Connect pair with a thin line
            ax.plot(
                [c_val, b_val], [y + y_off, y + y_off], color=col, alpha=0.6, zorder=2
            )

            # we don't add per-plot legend entries here; we'll create a
            # custom two-row legend after the loop so rows look consistent
            # across groups.
            pass

    # Y axis: labels
    ax.set_yticks(y_positions)
    # Make y-axis labels smaller and slanted to save horizontal space
    ax.set_yticklabels(groups, fontsize=11, rotation=45, ha="right")
    ax.set_xlim(xlim)
    ax.set_xlabel("Cosine similarity", fontsize=12)
    if title:
        ax.set_title(title)

    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # # Build custom legend: shapes (clean hollow) left-to-right, then
    # # 'clean encoder' text; second row same with filled shapes and
    # # 'backdoored encoder' text.
    # num_pairs = len(next(iter(data.values()))["clean"])
    # # In case marker_shapes length differs, cap by available shapes
    # num_pairs = min(num_pairs, len(marker_shapes))

    # handles_clean = [
    #     plt.Line2D(
    #         [0],
    #         [0],
    #         marker=marker_shapes[i],
    #         linestyle="None",
    #         markersize=8,
    #         markeredgecolor=colors[i],
    #         markerfacecolor="none",
    #     )
    #     for i in range(num_pairs)
    # ]
    # handle_clean_text = plt.Line2D([0], [0], linestyle="None", marker=None)

    # handles_bd = [
    #     plt.Line2D(
    #         [0],
    #         [0],
    #         marker=marker_shapes[i],
    #         linestyle="None",
    #         markersize=8,
    #         markeredgecolor=colors[i],
    #         markerfacecolor=colors[i],
    #     )
    #     for i in range(num_pairs)
    # ]
    # handle_bd_text = plt.Line2D([0], [0], linestyle="None", marker=None)

    # handles = handles_clean + [handle_clean_text] + handles_bd + [handle_bd_text]
    # labels = (
    #     [""] * num_pairs + ["clean encoder"] + [""] * num_pairs + ["backdoored encoder"]
    # )

    # ax.legend(
    #     handles=handles,
    #     labels=labels,
    #     ncol=(num_pairs + 1),
    #     handletextpad=0.08,
    #     columnspacing=0.15,
    #     frameon=False,
    #     loc="lower left",
    # )
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Example usage with the `data` defined above
    fig, ax = plot_grouped_pairs(data)
    plt.show()
