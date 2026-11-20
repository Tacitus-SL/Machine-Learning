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
        r"""Take definition of derivative and insert f(g(x))
instead of f(x):
$$\frac{d f(g(x))}{dx} = \frac{f(g(x + dx)) - f(g(x))}{dx}$$""",
        r"""To simplify the first term, take definition of
derivative
$$\frac{df}{dx} = \frac{f(x + dx) - f(x)}{dx}$$
And perform a small mutation:
$$f(x + dx) = f'(x) dx + f(x) + o(dx)$$
Using this expression, we can expand the first term""",
        r"""Simplify $g(x + dx)$:
$$\frac{f(g(x) + g'(x)dx) - f(g(x))}{dx}$$""",
        r"""Expand the first term yet another time with $y = g(x)$ and $dy = \frac{dg(x)}{dx} dx$:
$$\frac{f(y + dy) - f(g(x))}{dx} = \frac{f(y) + f'(y) dy - f(g(x))}{dx}$$""",
        r"""Returning g(x) instead of $y$ and $g'(x) dx$ instead of $dy$:
$$\frac{f(y) + f'(y) dy - f(g(x))}{dx} = \frac{f(g(x)) + f'(g(x)) g'(x) dx - f(g(x))}{dx}$$""",
        r"""The first and the last terms cancel out:
$$\frac{f'(g(x)) g'(x) dx}{dx}$$""",
        r"""$dx$ in the numerator and the denominator cancel out:
$$f'(g(x)) g'(x)$$""",
        r"""So, the derivative of compound function $f(g(x))$ is:
$$\frac{df(g(x))}{dx} = f'(g(x)) g'(x) = \frac{df}{dg}\frac{dg}{dx}$$""",
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
        # Derivative of Compound Function

        Order the cards to get the proof of the derivative
        of the compound function $f(g(x))$.
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
