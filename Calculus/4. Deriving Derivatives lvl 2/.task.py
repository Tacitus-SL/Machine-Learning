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
        r"""Take derivative of a function and insert $f(x) = \sin (x)$:
$$\frac{d f(x)}{dx} = \frac{f(x + dx) - f(x)}{dx} = \frac{\sin(x + dx) - \sin(x)}{dx}$$""",
        r"""Use formula of sinus of sum:
$$\sin(a + b) = \sin(a) \cos(b) + \cos(a) \sin(b)$$
to expand the first term in the numerator.""",
        r"""Expanding the first term in the numerator:
$$\frac{\sin(x) \cos(dx) + \cos(x) \sin(dx) - \sin(x)}{dx}$$""",
        r"""Firstly, for small values of $dx$ (The First Wonderful Limit): 
$$\sin(dx) \approx dx$$""",
        r"""Secondly, for small values of $dx$ (due to the same reason): 
$$\cos(dx) = \sqrt{1 - \sin^2 dx} \approx \sqrt{1 - dx^2} \approx 1 - \frac{dx^2}{2}$$""",
        r"""Substitute the approximations into the expression:
$$\frac{\sin(x) \left(1 - \frac{dx^2}{2}\right) + \cos(x) dx - \sin(x)}{dx}$$""",
        r"""Simplify the expression:
$$\frac{\sin(x) - \frac{dx^2}{2} \sin(x) + \cos(x) dx - \sin(x)}{dx}$$""",
        r"""The first and the last term cancels:
$$\frac{- \frac{dx^2}{2} \sin(x) + \cos(x) dx}{dx}$$""",
        r"""$dx$ in numerator and in denominator cancels:
$$- \frac{dx}{2} \sin(x) + \cos(x)$$""",
        r"""For small values of $dx$ first term is negligible:
$$\cos(x)$$""",
        r"""So, the derivative of $\sin(x)$ is $\cos(x)$:
$$\frac{d \sin(x)}{dx} = \cos(x)$$""",
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
        # Derivative of Sine

        Order the cards to get a proof of the derivative of sine
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
