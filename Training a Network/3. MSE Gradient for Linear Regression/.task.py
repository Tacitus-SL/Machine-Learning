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
        r"""Consider a linear regression model for $n$ samples and $d$ features.

The design matrix is $X \in \mathbb{R}^{n \times d}$, the weight vector is $w \in \mathbb{R}^{d}$, and the targets are $y \in \mathbb{R}^{n}$.

Predictions are collected into one vector:

$$\hat y = Xw$$""",
        r"""The mean squared error can be written compactly through the residual vector $r = Xw - y$:

$$L(w) = \frac{1}{n}\|Xw - y\|_2^2 = \frac{1}{n}r^T r$$""",
        r"""Substitute the residual definition into the quadratic form:

$$L(w) = \frac{1}{n}(Xw - y)^T(Xw - y)$$""",
        r"""Expand the product while remembering that the middle terms are scalars:

$$L(w) = \frac{1}{n}\left(w^T X^T Xw - w^T X^T y - y^T Xw + y^T y\right)$$""",
        r"""Since $w^T X^T y$ and $y^T Xw$ are the same scalar, combine them:

$$L(w) = \frac{1}{n}\left(w^T X^T Xw - 2w^T X^T y + y^T y\right)$$""",
        r"""Differentiate term by term with respect to $w$.

The target-only term $y^T y$ does not depend on $w$, and $X^T X$ is symmetric:

$$\nabla_w L(w) = \frac{1}{n}\left(2X^T Xw - 2X^T y\right)$$""",
        r"""Factor the common terms:

$$\nabla_w L(w) = \frac{2}{n}X^T(Xw - y)$$""",
        r"""The final expression has shape $\mathbb{R}^d$, matching the weight vector. Geometrically, $X^T$ maps prediction errors back into feature coordinates, telling each weight how it should change.

As you can see, the covariance matrix $X^T X$ plays an important role here.""",
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
        # MSE Gradient for Linear Regression

        Order the cards to derive the gradient of mean squared error for
        linear regression in matrix form.
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
