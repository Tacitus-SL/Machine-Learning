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

    FIXED_CARDS = np.array([
        r"""$$\frac{d}{dx}x^k$$""",
        r"""$$\frac{d}{dx}\exp(x)$$""",
        r"""$$\frac{d}{dx}\sin(x)$$""",
        r"""$$\frac{d}{dx}\cos(x)$$""",
        r"""$$\frac{d}{dx}\ln x, \; x > 0$$""",
        r"""$$\frac{d}{dx}\tg x, \; x \ne \frac{\pi}{2} + \pi n$$""",
        r"""$$\frac{d}{dx}\ctg x, \; x \ne \pi n$$""",
        r"""$$\frac{d}{dx}\arcsin x, \; |x| < 1$$""",
        r"""$$\frac{d}{dx}\arccos x, \; |x| < 1$$""",
        r"""$$\frac{d}{dx}\arctg x$$""",
        r"""$$\frac{d}{dx}\arcctg x$$""",
        r"""$$\frac{d}{dx}(f(x) + g(x))$$""",
        r"""$$\frac{d}{dx}(f(x) g(x))$$""",
        r"""$$\frac{d}{dx}(C f(x))$$""",
        r"""$$\frac{d}{dx}(f(g(x)))$$""",
    ])
    CORRECT_MATCHING_CARDS = np.array([
        r"""$$k x^{k-1}$$""",
        r"""$$\exp(x)$$""",
        r"""$$\cos(x)$$""",
        r"""$$-\sin(x)$$""",
        r"""$$x^{-1}$$""",
        r"""$$\frac{1}{\cos^2 x}$$""",
        r"""$$- \frac{1}{\sin^2 x}$$""",
        r"""$$\frac{1}{\sqrt{1 - x^2}}$$""",
        r"""$$- \frac{1}{\sqrt{1 - x^2}}$$""",
        r"""$$\frac{1}{1 + x^2}$$""",
        r"""$$- \frac{1}{1 + x^2}$$""",
        r"""$$\frac{df(x)}{dx} + \frac{dg(x)}{dx}$$""",
        r"""$$\frac{d f(x)}{dx} g(x) + f(x) \frac{dg(x)}{dx}$$""",
        r"""$$C \frac{d f(x)}{dx} $$""",
        r"""$$\frac{d f}{d g} \frac{d g}{d x}$$""",
    ])
    utils.cards.generate_matching_inputs(
        FIXED_CARDS,
        CORRECT_MATCHING_CARDS,
    )
    utils.checker.save_npz({'cards': CORRECT_MATCHING_CARDS}, utils.checker.get_data_dir() / '.correct_cards.npz')
    # -- end --

    FIXED_CARDS, MATCHING_CARDS = utils.cards.load_matching_cards()
    get_order, set_order = utils.cards.init_ordering_state(mo, MATCHING_CARDS)
    return (
        FIXED_CARDS,
        MATCHING_CARDS,
        get_order,
        mo,
        set_order,
        utils,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Match Functions with their Derivatives

        And memorize this table.

        This table has to be memorized. By heart. No excuses.
        """
    )
    return


@app.cell(hide_code=True)
def _(FIXED_CARDS, MATCHING_CARDS, get_order, mo, set_order, utils):
    order, widget, up_buttons, down_buttons = (
        utils.cards.build_matching_widget_marimo(
            mo,
            FIXED_CARDS,
            MATCHING_CARDS,
            get_order,
            set_order,
        )
    )
    utils.cards.save_matching_answer(MATCHING_CARDS, order)
    return down_buttons, up_buttons, widget


@app.cell(hide_code=True)
def _(down_buttons, up_buttons, utils, widget):
    # Referencing both arrays in another cell keeps their callbacks registered.
    _controls = down_buttons, up_buttons

    # -- start --
    print("Saving correct answer")
    _correct_matching_cards = utils.checker.load_answer(utils.checker.get_data_dir() / '.correct_cards.npz')['cards']
    utils.cards.save_matching_answer(_correct_matching_cards)
    # -- end --

    widget
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
