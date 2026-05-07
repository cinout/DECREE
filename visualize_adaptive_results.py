# adaptive_opt_1
# results = {
#     "alpha=0": {"acc": 52.6, "asr": 92.6, "s": 0.65},
#     "alpha=0.2": {"acc": 52.6, "asr": 49.6, "s": 0.60},
#     "alpha=0.5": {"acc": 51.3, "asr": 27.7, "s": 0.58},
#     "alpha=1.0": {"acc": 48.1, "asr": 12.8, "s": 0.53},
# }
# std = {
#     "alpha=0": {"acc": 10.5, "asr": 5.7, "s": 0.1},
#     "alpha=0.2": {"acc": 10.3, "asr": 20.7, "s": 0.1},
#     "alpha=0.5": {"acc": 9.7, "asr": 23.4, "s": 0.1},
#     "alpha=1.0": {"acc": 8.8, "asr": 13.0, "s": 0.1},
# }

# adaptive_opt_2
results = {
    "alpha=0": {"acc": 52.6, "asr": 92.6, "s": 0.65},
    # "alpha=0.2": {"acc": 50.5, "asr": 58.4, "s": 0.56},
    "alpha=0.2": {"acc": 50.5, "asr": 58.4, "s": 0.71},
    "alpha=0.5": {"acc": 50.8, "asr": 58.9, "s": 0.74},
    "alpha=1.0": {"acc": 47.1, "asr": 60.9, "s": 0.75},
}
std = {
    "alpha=0": {"acc": 10.5, "asr": 5.7, "s": 0.1},
    # "alpha=0.2": {"acc": 11.5, "asr": 45.8, "s": 0.44},
    "alpha=0.2": {"acc": 11.5, "asr": 45.8, "s": 0.06},
    "alpha=0.5": {"acc": 12.2, "asr": 44.2, "s": 0.05},
    "alpha=1.0": {"acc": 16.6, "asr": 40.3, "s": 0.13},
}


def plot_adaptive_results(results_dict, output_path=None, figsize=(8, 9)):
    import re
    import matplotlib.pyplot as plt

    # parse alpha floats from keys and sort
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

    # prepare std/error values if available in module-level `std` dict
    std_acc = None
    std_asr = None
    std_s = None
    std_dict = globals().get("std")
    if isinstance(std_dict, dict):
        std_map = {}
        for k, v in std_dict.items():
            m = re.search(r"[-+]?[0-9]*\.?[0-9]+", k)
            a = float(m.group()) if m else None
            std_map[a] = v
        std_acc = [std_map.get(a, 0).get("acc", 0) for a in alphas]
        std_asr = [std_map.get(a, 0).get("asr", 0) for a in alphas]
        std_s = [std_map.get(a, 0).get("s", 0) for a in alphas]

    # use numeric alphas for x-axis so line plots show trend
    x = alphas
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True, constrained_layout=True
    )

    # helper to format numeric labels
    def _fmt_values(vals):
        mx = max(vals) if vals else 1
        if mx <= 1:
            return [f"{v:.2f}" for v in vals]
        else:
            return [f"{v:.1f}" for v in vals]

    def _annotate_points(ax, xs, ys):
        labels = _fmt_values(ys)
        for xi, yi, lab in zip(xs, ys, labels):
            ax.text(xi, yi, lab, ha="center", va="bottom", fontsize=10)

    # ACC line
    axes[0].plot(x, acc, marker="o", color="mediumseagreen", linewidth=2)
    if std_acc is not None:
        lower = [a - b for a, b in zip(acc, std_acc)]
        upper = [a + b for a, b in zip(acc, std_acc)]
        axes[0].fill_between(x, lower, upper, color="mediumseagreen", alpha=0.15)
    axes[0].set_ylabel("ACC%", fontsize=11)
    axes[0].set_ylim(0, 100)
    _annotate_points(axes[0], x, acc)

    # ASR line
    axes[1].plot(x, asr, marker="o", color="indianred", linewidth=2)
    if std_asr is not None:
        lower = [a - b for a, b in zip(asr, std_asr)]
        upper = [a + b for a, b in zip(asr, std_asr)]
        axes[1].fill_between(x, lower, upper, color="indianred", alpha=0.12)
    axes[1].set_ylabel("ASR%", fontsize=11)
    axes[1].set_ylim(0, 100)
    _annotate_points(axes[1], x, asr)

    # s line
    axes[2].plot(x, s, marker="o", color="cornflowerblue", linewidth=2)
    if std_s is not None:
        lower = [a - b for a, b in zip(s, std_s)]
        upper = [a + b for a, b in zip(s, std_s)]
        axes[2].fill_between(x, lower, upper, color="cornflowerblue", alpha=0.12)
    axes[2].set_ylabel(r"$s$", fontsize=11)
    axes[2].set_ylim(0, 1)
    _annotate_points(axes[2], x, s)

    # x axis ticks and labels
    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[2].set_xlabel(r"$\alpha$", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)

    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    # Display the figure using the `results` dictionary above.
    plot_adaptive_results(results)
