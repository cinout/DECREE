# adaptive_opt_1
results_1 = {
    "alpha=0": {"acc": 52.6, "asr": 92.6, "s": 0.65},
    "alpha=0.2": {"acc": 52.6, "asr": 49.6, "s": 0.60},
    "alpha=0.5": {"acc": 51.3, "asr": 27.7, "s": 0.58},
    "alpha=1.0": {"acc": 48.1, "asr": 12.8, "s": 0.53},
}
std_1 = {
    "alpha=0": {"acc": 10.5, "asr": 5.7, "s": 0.1},
    "alpha=0.2": {"acc": 10.3, "asr": 20.7, "s": 0.1},
    "alpha=0.5": {"acc": 9.7, "asr": 23.4, "s": 0.1},
    "alpha=1.0": {"acc": 8.8, "asr": 13.0, "s": 0.1},
}

# # adaptive_opt_2
# results = {
#     "alpha=0": {"acc": 52.6, "asr": 92.6, "s": 0.65},
#     "alpha=0.2": {"acc": 50.5, "asr": 58.4, "s": 0.56},
#     # "alpha=0.2": {"acc": 50.5, "asr": 58.4, "s": 0.71},
#     "alpha=0.5": {"acc": 50.8, "asr": 58.9, "s": 0.74},
#     "alpha=1.0": {"acc": 47.1, "asr": 60.9, "s": 0.75},
# }
# std = {
#     "alpha=0": {"acc": 10.5, "asr": 5.7, "s": 0.1},
#     "alpha=0.2": {"acc": 11.5, "asr": 45.8, "s": 0.44},
#     # "alpha=0.2": {"acc": 11.5, "asr": 45.8, "s": 0.06},
#     "alpha=0.5": {"acc": 12.2, "asr": 44.2, "s": 0.05},
#     "alpha=1.0": {"acc": 16.6, "asr": 40.3, "s": 0.13},
# }

# adaptive_opt_3
results_2 = {
    "alpha=0": {"acc": 52.6, "asr": 92.6, "s": 0.65},
    # "alpha=0.2": {"acc": 50.5, "asr": 58.4, "s": 0.56},
    "alpha=0.2": {"acc": 51.4, "asr": 81.2, "s": 0.59},
    "alpha=0.5": {"acc": 51.2, "asr": 86.4, "s": 0.57},
    "alpha=1.0": {"acc": 48.8, "asr": 85.5, "s": 0.64},
}
std_2 = {
    "alpha=0": {"acc": 10.5, "asr": 5.7, "s": 0.1},
    # "alpha=0.2": {"acc": 11.5, "asr": 45.8, "s": 0.44},
    "alpha=0.2": {"acc": 9.9, "asr": 23.9, "s": 0.19},
    "alpha=0.5": {"acc": 10.9, "asr": 29.5, "s": 0.45},
    "alpha=1.0": {"acc": 11.7, "asr": 29.4, "s": 0.21},
}


def plot_adaptive_results(
    results1,
    std1=None,
    results2=None,
    std2=None,
    output_path=None,
    figsize=(12, 9),
):
    import re
    import matplotlib.pyplot as plt

    def _parse(results_dict):
        items = []
        for k, v in results_dict.items():
            m = re.search(r"[-+]?[0-9]*\.?[0-9]+", k)
            alpha = float(m.group()) if m else None
            items.append((alpha, v))
        items = sorted(items, key=lambda x: x[0])
        alphas = [a for a, _ in items]
        labels = [str(a) for a in alphas]
        acc = [vals["acc"] for _, vals in items]
        asr = [vals["asr"] for _, vals in items]
        s = [vals["s"] for _, vals in items]
        return alphas, labels, acc, asr, s

    def _parse_std(std_dict, alphas):
        if not isinstance(std_dict, dict):
            return None, None, None
        std_map = {}
        for k, v in std_dict.items():
            m = re.search(r"[-+]?[0-9]*\.?[0-9]+", k)
            a = float(m.group()) if m else None
            std_map[a] = v
        std_acc = [std_map.get(a, {}).get("acc", 0) for a in alphas]
        std_asr = [std_map.get(a, {}).get("asr", 0) for a in alphas]
        std_s = [std_map.get(a, {}).get("s", 0) for a in alphas]
        return std_acc, std_asr, std_s

    a1, labels1, acc1, asr1, s1 = _parse(results1)
    std_acc1, std_asr1, std_s1 = _parse_std(
        std1 if std1 is not None else globals().get("std"), a1
    )

    has_second = results2 is not None
    if has_second:
        a2, labels2, acc2, asr2, s2 = _parse(results2)
        std_acc2, std_asr2, std_s2 = _parse_std(
            std2 if std2 is not None else globals().get("std2"), a2
        )

    # plotting layout: 3 rows x (1 or 2) cols
    ncols = 2 if has_second else 1
    fig, axes = plt.subplots(
        3, ncols, figsize=figsize, sharey=False, constrained_layout=True
    )
    if ncols == 1:
        axes = axes[:, None]

    def _fmt_values(vals):
        mx = max(vals) if vals else 1
        if mx <= 1:
            return [f"{v:.2f}" for v in vals]
        else:
            return [f"{v:.1f}" for v in vals]

    def _annotate_points(ax, xs, ys, offset=(0, 0)):
        labels = _fmt_values(ys)
        for xi, yi, lab in zip(xs, ys, labels):
            ax.text(
                xi + offset[0],
                yi + offset[1],
                lab,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # ACC
    ax = axes[0, 0]
    ax.plot(a1, acc1, marker="o", color="mediumseagreen", linewidth=2)
    if std_acc1 is not None:
        lower = [a - b for a, b in zip(acc1, std_acc1)]
        upper = [a + b for a, b in zip(acc1, std_acc1)]
        ax.fill_between(a1, lower, upper, color="mediumseagreen", alpha=0.15)
    ax.set_ylabel("ACC%", fontsize=13)
    ax.set_ylim(0, 100)
    _annotate_points(ax, a1, acc1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if has_second:
        ax2 = axes[0, 1]
        ax2.plot(a2, acc2, marker="o", color="mediumseagreen", linewidth=2)
        if std_acc2 is not None:
            lower = [a - b for a, b in zip(acc2, std_acc2)]
            upper = [a + b for a, b in zip(acc2, std_acc2)]
            ax2.fill_between(a2, lower, upper, color="mediumseagreen", alpha=0.15)
        ax2.set_ylim(0, 100)
        _annotate_points(ax2, a2, acc2)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # ASR
    ax = axes[1, 0]
    ax.plot(a1, asr1, marker="o", color="indianred", linewidth=2)
    if std_asr1 is not None:
        lower = [a - b for a, b in zip(asr1, std_asr1)]
        upper = [a + b for a, b in zip(asr1, std_asr1)]
        ax.fill_between(a1, lower, upper, color="indianred", alpha=0.12)
    ax.set_ylabel("ASR%", fontsize=13)
    ax.set_ylim(0, 100)
    _annotate_points(ax, a1, asr1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if has_second:
        ax2 = axes[1, 1]
        ax2.plot(a2, asr2, marker="o", color="indianred", linewidth=2)
        if std_asr2 is not None:
            lower = [a - b for a, b in zip(asr2, std_asr2)]
            upper = [a + b for a, b in zip(asr2, std_asr2)]
            ax2.fill_between(a2, lower, upper, color="indianred", alpha=0.12)
        ax2.set_ylim(0, 100)
        _annotate_points(ax2, a2, asr2)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # s
    ax = axes[2, 0]
    ax.plot(a1, s1, marker="o", color="cornflowerblue", linewidth=2)
    if std_s1 is not None:
        lower = [a - b for a, b in zip(s1, std_s1)]
        upper = [a + b for a, b in zip(s1, std_s1)]
        ax.fill_between(a1, lower, upper, color="cornflowerblue", alpha=0.12)
    ax.set_ylabel(r"$s$", fontsize=13)
    ax.set_ylim(0, 1)
    _annotate_points(ax, a1, s1)
    ax.set_xlabel(r"$\alpha$", fontsize=13)
    ax.set_xticks(a1)
    ax.set_xticklabels(labels1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # place column titles below the bottom plots (centered under each column)
    bottom_left = axes[2, 0]
    bottom_left.text(
        0.5,
        -0.38,
        r"(a) $L_{bypass}=L_{trigger}$",
        ha="center",
        va="top",
        transform=bottom_left.transAxes,
        fontsize=13,
    )

    if has_second:
        bottom_right = axes[2, 1]
        bottom_right.text(
            0.5,
            -0.38,
            r"(b) $L_{bypass}=-s$",
            ha="center",
            va="top",
            transform=bottom_right.transAxes,
            fontsize=13,
        )

    if has_second:
        ax2 = axes[2, 1]
        ax2.plot(a2, s2, marker="o", color="cornflowerblue", linewidth=2)
        if std_s2 is not None:
            lower = [a - b for a, b in zip(s2, std_s2)]
            upper = [a + b for a, b in zip(s2, std_s2)]
            ax2.fill_between(a2, lower, upper, color="cornflowerblue", alpha=0.12)
        ax2.set_ylim(0, 1)
        _annotate_points(ax2, a2, s2)
        ax2.set_xlabel(r"$\alpha$", fontsize=13)
        ax2.set_xticks(a2)
        ax2.set_xticklabels(labels2)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    # Display the figure using the `results` dictionary above.
    # left: results_1/std_1, right: results_2/std_2
    plot_adaptive_results(results_1, std_1, results_2, std_2)
