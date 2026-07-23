"""Visualize a trained (or untrained) Keras Sequential/Dense network
as a node-and-edge diagram.

Usage in a notebook:
    from nn_viz import draw_network
    draw_network(model)                       # everything inferred
    draw_network(model, node_labels=[...])    # label the input nodes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _extract_dense_params(model):
    """Pull (weights, biases, activations) from every Dense-like layer.

    Duck-typed on get_weights() returning [W (2-D), b (1-D)], so it works
    across Keras/TF versions and ignores Input, Dropout, Flatten, etc.
    """
    Ws, bs, acts = [], [], []
    for layer in model.layers:
        params = layer.get_weights() if hasattr(layer, "get_weights") else []
        if len(params) == 2 and params[0].ndim == 2 and params[1].ndim == 1:
            Ws.append(params[0])
            bs.append(params[1])
            act = getattr(layer, "activation", None)
            acts.append(getattr(act, "__name__", "") if act else "")
    if not Ws:
        raise ValueError("No Dense-style layers with weights found in model.")
    # sanity-check that consecutive layers chain together
    for W_prev, W_next in zip(Ws, Ws[1:]):
        if W_prev.shape[1] != W_next.shape[0]:
            raise ValueError("Layers don't chain; is this a plain "
                             "sequential dense network?")
    return Ws, bs, acts


def draw_network(model, node_labels=None, output_labels=None,
                 layer_labels=None, show_biases=True, figsize=(10, 8),
                 max_lw=2.5, ax=None):
    """Draw a Keras dense network as nodes and edges.

    model         : a Keras model built from Dense layers. Layer sizes,
                    weights, biases, and activations are read from it.
    node_labels   : optional list of strings for the input-layer nodes,
                    drawn to the left.
    output_labels : optional list of strings (or a single string) for
                    the output-layer nodes, drawn to the right.
    layer_labels  : optional list of strings, one per layer; if omitted,
                    labels are auto-generated from sizes and activations.
    show_biases   : if True, node fill color encodes bias sign/magnitude
                    (input nodes, which have no bias, stay white).
    """
    if isinstance(output_labels, str):
        output_labels = [output_labels]
    Ws, bs, acts = _extract_dense_params(model)
    layer_sizes = [Ws[0].shape[0]] + [W.shape[1] for W in Ws]
    n_layers = len(layer_sizes)

    if layer_labels is None:
        layer_labels = [f"input\n({layer_sizes[0]} features)"]
        for i, (n, a) in enumerate(zip(layer_sizes[1:], acts)):
            kind = "output" if i == len(acts) - 1 else "hidden"
            layer_labels.append(f"{kind}\n({n}, {a})" if a else f"{kind}\n({n})")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_positions = np.linspace(0, 1, n_layers)
    y_positions = [np.linspace(0, 1, n + 2)[1:-1] if n > 1 else np.array([0.5])
                   for n in layer_sizes]

    # --- edges (drawn first, so nodes sit on top) ---
    for i, W in enumerate(Ws):
        wmax = np.abs(W).max() or 1.0
        for a, ya in enumerate(y_positions[i]):
            for b, yb in enumerate(y_positions[i + 1]):
                w = W[a, b]
                strength = abs(w) / wmax
                ax.add_line(Line2D(
                    [x_positions[i], x_positions[i + 1]], [ya, yb],
                    color='tab:blue' if w >= 0 else 'tab:red',
                    lw=0.2 + max_lw * strength,
                    alpha=0.05 + 0.75 * strength, zorder=1))

    # --- nodes ---
    node_size = max(60, 3000 / max(layer_sizes))
    bmax = max((np.abs(b).max() for b in bs), default=1.0) or 1.0
    for i, ys in enumerate(y_positions):
        if show_biases and i > 0:
            b = bs[i - 1]
            # map bias to a red-white-blue face color
            colors = plt.cm.coolwarm_r(0.5 + 0.5 * b / bmax)
        else:
            colors = ['white'] * len(ys)
        ax.scatter([x_positions[i]] * len(ys), ys, s=node_size,
                   facecolor=colors, edgecolor='black', zorder=2)

    # --- labels & legend ---
    if node_labels is not None:
        for lbl, y in zip(node_labels, y_positions[0]):
            ax.text(-0.03, y, lbl, ha='right', va='center', fontsize=10)
    if output_labels is not None:
        for lbl, y in zip(output_labels, y_positions[-1]):
            ax.text(1.03, y, lbl, ha='left', va='center', fontsize=10)
    for lbl, x in zip(layer_labels, x_positions):
        ax.text(x, -0.06, lbl, ha='center', va='top', fontsize=11)

    handles = [Line2D([0], [0], color='tab:blue', lw=2),
               Line2D([0], [0], color='tab:red', lw=2)]
    names = ['positive weight', 'negative weight']
    if show_biases:
        handles += [Line2D([0], [0], marker='o', color='none', mec='black',
                           mfc=plt.cm.coolwarm_r(0.9), ms=8),
                    Line2D([0], [0], marker='o', color='none', mec='black',
                           mfc=plt.cm.coolwarm_r(0.1), ms=8)]
        names += ['positive bias', 'negative bias']
    ax.legend(handles, names, loc='upper right', frameon=False, fontsize=9)

    ax.set_xlim(-0.25, 1.3 if output_labels is not None else 1.1)
    ax.set_ylim(-0.12, 1.05)
    ax.axis('off')
    fig.tight_layout()
    return fig, ax
