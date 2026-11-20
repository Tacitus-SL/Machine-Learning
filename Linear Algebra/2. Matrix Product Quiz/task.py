import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import importlib
    import os
    import sys
    from pathlib import Path

    import marimo as mo

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import utils

    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]

    importlib.invalidate_caches()
    utils = importlib.import_module("utils")


    CARDS = utils.cards.load_ordering_cards()
    get_selected, set_selected = utils.cards.init_selection_state(mo, CARDS)


    return CARDS, get_selected, mo, set_selected, utils


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Matrix Product Quiz

        Select the correct statements.
        """
    )
    return


@app.cell(hide_code=True)
def _(CARDS, get_selected, mo, set_selected, utils):
    selected, widget, checkboxes = (
        utils.cards.build_selection_widget_marimo(
            mo,
            CARDS,
            get_selected,
            set_selected,
        )
    )
    utils.cards.save_selection_answer(CARDS, selected)
    return checkboxes, widget


@app.cell(hide_code=True)
def _(checkboxes, utils, widget):
    # Referencing the array in another cell keeps its callbacks registered.
    _controls = checkboxes


    widget
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
