results = {
    "alpha=0": {"acc": 52.6, "asr": 92.6, "s": 0.65},
    "alpha=0.2": {"acc": 52.6, "asr": 49.6, "s": 0.60},
    "alpha=0.5": {"acc": 51.3, "asr": 27.7, "s": 0.58},
    "alpha=1.0": {"acc": 48.1, "asr": 12.8, "s": 0.53},
}
std = {
    "alpha=0": {"acc": 10.5, "asr": 5.7, "s": 0.1},
    "alpha=0.2": {"acc": 10.3, "asr": 20.7, "s": 0.1},
    "alpha=0.5": {"acc": 9.7, "asr": 23.4, "s": 0.1},
    "alpha=1.0": {"acc": 8.8, "asr": 13.0, "s": 0.1},
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

    x = range(len(alphas))
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True, constrained_layout=True
    )

    if std_acc is not None:
        bars0 = axes[0].bar(
            x,
            acc,
            color="mediumseagreen",
            width=0.6,
            yerr=std_acc,
            capsize=6,
            error_kw={"ecolor": "black", "lw": 1},
        )
    else:
        bars0 = axes[0].bar(x, acc, color="mediumseagreen", width=0.6)
    axes[0].set_ylabel("ACC%", fontsize=11)
    # axes[0].set_title("Accuracy vs Alpha")

    if std_asr is not None:
        bars1 = axes[1].bar(
            x,
            asr,
            color="indianred",
            width=0.6,
            yerr=std_asr,
            capsize=6,
            error_kw={"ecolor": "black", "lw": 1},
        )
    else:
        bars1 = axes[1].bar(x, asr, color="indianred", width=0.6)
    axes[1].set_ylabel("ASR%", fontsize=11)
    # axes[1].set_title("Attack Success Rate vs Alpha")

    if std_s is not None:
        bars2 = axes[2].bar(
            x,
            s,
            color="cornflowerblue",
            width=0.6,
            yerr=std_s,
            capsize=6,
            error_kw={"ecolor": "black", "lw": 1},
        )
    else:
        bars2 = axes[2].bar(x, s, color="cornflowerblue", width=0.6)
    axes[2].set_ylabel(r"$s$", fontsize=11)
    # axes[2].set_title("S vs Alpha")

    # show alpha tick labels on every subplot
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.xaxis.set_tick_params(labelbottom=True)
    axes[2].set_xlabel(r"$\alpha$", fontsize=11)

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # annotate bars with numeric values
    def _fmt_values(vals):
        mx = max(vals) if vals else 1
        if mx <= 1:
            return [f"{v:.2f}" for v in vals]
        else:
            return [f"{v:.1f}" for v in vals]

    def _annotate(ax, bars, vals):
        labels = _fmt_values(vals)
        mx = max(vals) if vals else 1
        for bar, lab in zip(bars, labels):
            x_pos = bar.get_x() + bar.get_width() / 2
            h = bar.get_height()
            # place inside the bar (centered vertically)
            y_pos = h * 0.5
            # choose text color for contrast
            color = "white"
            # color = "white" if (h / mx) >= 0.25 else "black"
            ax.text(
                x_pos,
                y_pos,
                lab,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
            )

    _annotate(axes[0], bars0, acc)
    _annotate(axes[1], bars1, asr)
    _annotate(axes[2], bars2, s)

    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    # Display the figure using the `results` dictionary above.
    plot_adaptive_results(results)
