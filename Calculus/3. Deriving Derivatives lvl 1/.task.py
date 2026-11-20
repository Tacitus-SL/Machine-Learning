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
        r"""Start with the definition of derivative of a function and insert power function:

$$\frac{df}{dx} = \frac{f(x + dx) - f(x)}{dx}= 
\frac{(x + dx)^k - x^k}{dx}$$""",
        r"""Open first bracket:
$$\frac{x^k + k x^{k - 1} dx + \frac{k (k - 1)}{2} x^{k - 2} (dx)^2 + \dots + (dx)^k - x^k}{dx}$$""",
        r"""First and last terms cancel out:
$$\frac{k x^{k - 1} dx + \frac{k (k - 1)}{2} x^{k - 2} (dx)^2 + \dots + (dx)^k}{dx}$$""",
        r"""$dx$ in denominator cancels out:
$$k x^{k - 1} + \frac{k (k - 1)}{2} x^{k - 2} dx + \dots + (dx)^{k - 1}$$""",
        r"""Due to infinitesimality of $dx$ all terms except first one are negligible:
$$k x^{k - 1}$$""",
        r"""So, the derivative of power function is:
$$\frac{d x^k}{dx} = k x^{k - 1}$$""",
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
        # Derivative of Power Function

        Order the cards to get a proof of the derivative of the power function
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
