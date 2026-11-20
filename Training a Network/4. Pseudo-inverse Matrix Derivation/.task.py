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

    # -- start --
    import numpy as np

    ORDERED_CARDS = np.array([
        r"""Start from the same linear regression setup: $X \in \mathbb{R}^{n \times d}$, $w \in \mathbb{R}^{d}$, and $y \in \mathbb{R}^{n}$.

We want weights whose predictions $Xw$ minimize $\|Xw - y\|_2^2$.""",
        r"""From the MSE gradient derivation, the gradient is:

$$\nabla_w L(w) = \frac{2}{n}X^T(Xw - y)$$""",
        r"""At an optimum of this convex quadratic loss, set the gradient equal to zero:

$$\frac{2}{n}X^T(Xw - y) = 0$$""",
        r"""Remove the nonzero scalar factor and distribute $X^T$:

$$X^T Xw - X^T y = 0$$""",
        r"""Rearrange the equation into the normal equations:

$$X^T Xw = X^T y$$""",
        r"""If the columns of $X$ are linearly independent, then $X^T X$ is invertible. Multiply both sides by $(X^T X)^{-1}$:

$$w = (X^T X)^{-1}X^T y$$""",
        r"""The matrix multiplying $y$ is called the left pseudo-inverse of $X$:

$$X^+ = (X^T X)^{-1}X^T$$""",
        r"""Therefore the least-squares solution can be written as:

$$w = X^+y$$

This formula projects the target vector onto the column space of $X$ and chooses weights that produce the closest reachable prediction.

As one can see, $X^T X$ here is also quite important.""",
    ])
    utils.cards.generate_ordering_inputs(ORDERED_CARDS)
    utils.checker.save_npz({'cards': ORDERED_CARDS}, utils.checker.get_data_dir() / '.correct_cards.npz')
    # -- end --

    CARDS = utils.cards.load_ordering_cards()
    get_order, set_order = utils.cards.init_ordering_state(mo, CARDS)
    return CARDS, get_order, mo, set_order, utils


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Pseudo-inverse Matrix Derivation

        Order the cards to derive the pseudo-inverse solution of linear
        regression from the MSE gradient.
        """
    )
    return


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
def _(down_buttons, up_buttons, utils, widget):
    # Referencing both arrays in another cell keeps their callbacks registered.
    _controls = down_buttons, up_buttons

    # -- start --
    _ordered_cards = utils.checker.load_answer(utils.checker.get_data_dir() / '.correct_cards.npz')['cards']
    utils.cards.save_ordering_answer(_ordered_cards)
    # -- end --

    widget
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
