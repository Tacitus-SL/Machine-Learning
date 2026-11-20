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
        r"""Start with definition of $\mathbb V \xi$,
$$\mathbb V \xi = \mathbb E (\xi - \mathbb E \xi)^2$$""",
            r"""Opening the brackets and thinking about $\mathbb E$ as about the average, we will get:
$$\mathbb V \xi = \mathbb E (\xi^2 - 2 \xi \mathbb E \xi + (\mathbb E \xi)^2)$$""",
            r"""In fact, $\mathbb E \xi$ is a constant (as it is kind of average value), thus:
$$\mathbb V \xi = \mathbb E \xi^2 - 2 (\mathbb E \xi)^2 + (\mathbb E \xi)^2$$""",
            r"""Simplifying the expression above, we will get:
$$\mathbb V \xi = \mathbb E \xi^2 - (\mathbb E \xi)^2$$""",
            r"""Note that this expression is correct for any $\xi$ !!!""",
            r"""Now consider $\mathbb V (a \xi + b \eta)$""",
            r"""Take the formula that was proven above, and we will get:
$$\mathbb V (a \xi + b \eta) =\mathbb E (a \xi + b \eta)^2 - (\mathbb E (a \xi + b \eta))^2$$""",
            r"""Opening the brackets, we will get:
$$\mathbb V (a \xi + b \eta) =\mathbb E (a^2 \xi^2) + \mathbb E (2 a \xi b \eta) + \mathbb E (b \eta)^2 - (\mathbb E (a \xi))^2 - 2 \mathbb E a \xi \mathbb E b \eta - (\mathbb E (b \eta))^2$$""",
            r"""Taking out the coefficients, we get:
$$\mathbb V (a \xi + b \eta) = a^2 \mathbb E \xi^2 +2 a b \mathbb E  (\xi \eta) + b^2 \mathbb E \eta^2 - a^2 (\mathbb E \xi)^2 - 2 a b \mathbb E \xi \mathbb E \eta - b^2(\mathbb E \eta)^2$$""",
            r"""Recalling that $\mathbb V \xi = \mathbb E \xi^2 - (\mathbb E \xi)^2$:
$$\mathbb V (a \xi + b \eta) = a^2 \mathbb V \xi + b^2 \mathbb V \eta + 2 a b \mathbb E  (\xi \eta) - 2 a b \mathbb E \xi \mathbb E \eta$$""",
            r"""If we have independent $\xi$ and $\eta$, then
$$\mathbb E \xi \eta = \sum_{k} x_k y_k p(x_k, y_k) = \sum_{k} x_k y_k p(x_k)p(y_k) = \sum_{k} x_k p(x_k) \sum_{k} y_k p(y_k) = \mathbb E \xi \mathbb E \eta$$""",
            r"""In that case the two last terms are cancelled, and we get:
$$\mathbb V (a \xi + b \eta) = a^2 \mathbb V \xi + b^2 \mathbb V \eta$$""",
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
