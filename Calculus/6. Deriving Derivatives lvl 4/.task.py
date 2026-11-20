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
        r"""Consider combination of forward and inverse functions
$$\arcsin(\sin(x)) = x$$ 
(assuming $x \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$)""",
        r"""Take derivative of both sides:
$$\frac{d}{dx}\arcsin(\sin(x)) = \frac{dx}{dx}$$""",
        r"""Right side is easy:
$$\frac{dx}{dx} = 1$$
And thus:
$$\frac{d}{dx}\arcsin(\sin(x)) = 1$$""",
        r"""For the left side, use chain rule:
$$\frac{d}{dx}\arcsin(\sin(x)) = \frac{d \arcsin(\sin(x))}{d(\sin(x))}\frac{d(\sin(x))}{dx}$$""",
        r"""Replace $\sin x$ with $y$:
$$\frac{d}{dx}\arcsin(\sin(x)) = \frac{d \arcsin(y)}{d(y)}\frac{d(y)}{dx}$$""",
        r"""But this expression should be equal to 1!:
$$\frac{d \arcsin(y)}{d(y)}\frac{d(y)}{dx} = 1$$""",
        r"""So, we have:
$$\frac{d \arcsin(y)}{d(y)} = \frac{1}{\frac{dy}{dx}}$$""",
        r"""And we know that $\frac{dy}{dx} = \cos x$:
$$\frac{d \arcsin(y)}{d(y)} = \frac{1}{\cos x}$$""",
        r"""Finally, replace $\cos x$ with $\sqrt{1 - \sin^2 x}$:
$$\frac{d \arcsin(y)}{d(y)} = \frac{1}{\sqrt{1 - y^2}}$$""",
        r"""Hence, the derivative of $\arcsin x$ is:
$$\frac{d}{dx}\arcsin x = \frac{1}{\sqrt{1 - x^2}}$$""",
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
        # Derivative of arcsin

        Order the cards to get a proof of the derivative of $\arcsin x$
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
