import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():

    import importlib
    import sys
    import marimo as mo
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import utils

    for module_name in list(sys.modules):
      if module_name == "utils" or module_name.startswith("utils."):
          del sys.modules[module_name]

    importlib.invalidate_caches()
    utils = importlib.import_module("utils")

    # ORDERED_CARDS is the correct answer -- placeholder so this cell still
    # returns something defined if the marker block below is stripped
    # (student-facing task.py); the real value is only ever set inside it.

    # -- start --
    import numpy as np

    ORDERED_CARDS = np.array([
        r"""Start with definition of $\mathbb E \xi$,
$$\mathbb E \xi = \sum_k p(\xi=x_k) x_k,$$
where k runs over all possible $x_k$ outputs""",
            r"""Using the classical definition of probability, we may think of $p(\xi = x_k)$ as
about a fraction of the experiments that yield $x_k$ as the output""",
            r"""Thus, we can rewrite
$$\mathbb E \xi = \sum_k p(\xi=x_k) x_k = \sum_k \frac{\#(\xi = x_k)}{N} x_k = \frac{1}{N} \sum_{n = 1}^N \xi_n,$$
where $\xi_n$ is the $n$-th instance of random value $\xi$ and $n$ runs through all of
the $N$ experiments""",
            r"""Consider $\mathbb E (a \xi + b \eta)$ and rewrite as average of many experiments:""",
            r"""$$\mathbb E (a \xi + b \eta) = \frac{1}{N}\sum_{n=1}^N (a \xi_n + b \eta_n)$$""",
            r"""Rewriting this formula, we will get:
$$\mathbb E (a \xi + b \eta) = a \frac{1}{N}\sum_{n=1}^N \xi_n + b \frac{1}{N}\sum_{n=1}^N \eta_n$$""",
            r"""But we know that
$$\mathbb E \xi =\frac{1}{N}\sum_{n=1}^N \xi_n$$
$$\mathbb E \eta =\frac{1}{N}\sum_{n=1}^N \eta_n$$""",
            r"""Thus, we have proved:
$$\mathbb E (a \xi + b \eta) = a \mathbb E \xi + b \mathbb E \eta$$""",
        ])
    utils.cards.generate_ordering_inputs(ORDERED_CARDS)
    # -- end --

    CARDS = utils.cards.load_ordering_cards()
    get_order, set_order = utils.cards.init_ordering_state(mo, CARDS)
    return CARDS, ORDERED_CARDS, get_order, mo, set_order, utils


@app.cell(hide_code=True)
def _(CARDS, get_order, mo, set_order, utils):
    order, widget, up_buttons, down_buttons = (
        utils.cards.build_ordering_widget_marimo(
            mo,
            CARDS,
            get_order,
            set_order,
        )
    )
    utils.cards.save_ordering_answer(CARDS, order)
    return down_buttons, up_buttons, widget


@app.cell(hide_code=True)
def _(ORDERED_CARDS, down_buttons, up_buttons, utils, widget):
    # Referencing both arrays in another cell keeps their callbacks registered.
    _controls = down_buttons, up_buttons

    # -- start --
    utils.cards.save_ordering_answer(ORDERED_CARDS)
    # -- end --

    widget
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
