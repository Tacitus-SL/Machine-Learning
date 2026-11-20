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


    widget
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
