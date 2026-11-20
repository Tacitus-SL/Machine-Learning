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

    # -- start --
    import numpy as np

    STATEMENTS = np.array([
        r"""$\forall A \in \mathbb R^{N \times N}$ and
$\forall \mathbf x \in \mathbb R^M \Rightarrow
A \mathbf x \in \mathbb R^N$""",
        r"""$\forall A \in \mathbb R^{N \times N}$ and
$\forall \mathbf x \in \mathbb R^N \Rightarrow
A \mathbf x \in \mathbb R^N$""",
        r"""$\forall A \in \mathbb R^{N \times K}$ and
$\forall B \in \mathbb R^{K \times M} \Rightarrow
A B \in \mathbb R^{N \times M}$""",
        r"""$\forall A \in \mathbb R^{N \times N}$ and
$\forall B \in \mathbb R^{K \times M} \Rightarrow
A B \in \mathbb R^{N \times M}$""",
        r"""$\forall A = \begin{bmatrix}
  \mathbf a_1 &
  \dots &
  \mathbf a_N
\end{bmatrix} \in \mathbb R^{K \times N}$

and

$\forall B = \begin{bmatrix}
  \mathbf b_1 & \dots & \mathbf b_M
\end{bmatrix} \in \mathbb R^{K \times M} \Rightarrow$

$A^T B = \begin{bmatrix}
  \left<\mathbf a_1, \mathbf b_1\right> & \dots & \left<\mathbf a_1, \mathbf b_M\right> \\
  \vdots & \ddots & \vdots \\
  \left<\mathbf a_N, \mathbf b_1\right> & \dots & \left<\mathbf a_N, \mathbf b_M\right>
\end{bmatrix} \in \mathbb R^{N \times M}$""",
        r"""$\forall A \in \mathbb R^{K \times N}$ and
$\forall B = \begin{bmatrix}
  \mathbf b_1 & \dots & \mathbf b_M
\end{bmatrix} \in \mathbb R^{K \times M} \Rightarrow$
$A^T B = \begin{bmatrix}
  A^T \mathbf b_1 & \dots & A^T \mathbf b_M
\end{bmatrix} \in \mathbb R^{N \times M}$""",
        r"""$\forall B \in \mathbb R^{K \times M}$ and
$\forall A = \begin{bmatrix}
  \mathbf a_1 & \dots & \mathbf a_N
\end{bmatrix} \in \mathbb R^{K \times N} \Rightarrow$
$A^T B = \begin{bmatrix}
  \mathbf a_1^T B \\ \vdots \\ \mathbf a_N^T B
\end{bmatrix} \in \mathbb R^{N \times M}$""",
        r"""Suppose that we want to calculate the product $A B$ and we realise
that matrix $A$ contains zeros in $50\%$ of the positions. That
means that we can perform $50\%$ less operations of multiplication to
calculate $AB$""",
        r"""Suppose that we want to calculate the product $A B$ and we realise
that both matrices $A$ and $B$ contains zeros in $50\%$ of the
positions. That
means that we can perform $75\%$ less operations of multiplication to
calculate $AB$""",
        r"""Matrix product $AB$ is always equal to elementwise matrix
product $A \cdot B$""",
        r"""$\forall \alpha, \beta \in \mathbb R$, $\forall \mathbf x \in \mathbb R^M$ and
$\forall A, B \in \mathbb R^{N \times M} \Rightarrow$
$(\alpha A + \beta B ) \mathbf x = \alpha A \mathbf x + \beta B \mathbf x$""",
        r"""$\forall \alpha, \beta \in \mathbb R$,
$\forall \mathbf x, \mathbf y \in \mathbb R^M$ and
$\forall A \in \mathbb R^{N \times M} \Rightarrow$
$A (\alpha \mathbf x + \beta \mathbf y)  = \alpha A \mathbf x + \beta A \mathbf y$""",
    ])
    CORRECT = np.array([False, True, True, False, True, True, True, True, False, False, True, True])

    # Reused as-is from the ordering widget: shuffling display order and
    # detecting stale card text is identical work here, and the shuffle
    # itself does not leak which statements are correct.
    utils.cards.generate_ordering_inputs(STATEMENTS)
    utils.checker.save_npz(
        {'cards': STATEMENTS, 'correct': CORRECT},
        utils.checker.get_data_dir() / '.correct_selection.npz',
    )
    # -- end --

    CARDS = utils.cards.load_ordering_cards()
    get_selected, set_selected = utils.cards.init_selection_state(mo, CARDS)

    # -- start --
    # Author-preview convenience: compile.py copies this file to task.py
    # as a raw, unstripped copy -- opening/running THAT copy (before
    # cleanup.py runs) hits this block. It never touches the real
    # state.npz (already loaded, untouched, by init_selection_state()
    # above) -- instead it takes the correct selection, round-trips it
    # through a throwaway file in the checker's own hidden data dir,
    # loads that back into a plain state variable, deletes the throwaway
    # file, and only then pushes that variable into the live widget
    # state via set_selected(). This whole block disappears from task.py
    # once cleanup.py strips it, so a published task's widget always
    # starts from whatever init_selection_state() loaded, unmodified.
    #
    # Guarded by CHECKER_DATA_DIR: an unconditional version would also
    # fire during .task.py's own runs inside a real check_code_task()
    # grading pass (it is always the teacher script there), pushing the
    # correct selection into the live widget state before the student's
    # task.py ever runs. Since save_selection_answer() below reads the
    # live widget value (not the file), that would leak into the
    # teacher's own answer -- harmless by itself here since the override
    # cell forces the correct answer anyway, but the guard keeps this
    # block's behavior consistent with the other cards tasks and avoids
    # relying on that coincidence.
    if not os.environ.get('CHECKER_DATA_DIR'):
        _correct = utils.checker.load_answer(
            utils.checker.get_data_dir() / '.correct_selection.npz'
        )
        _correct_by_text = dict(zip(_correct['cards'].tolist(), _correct['correct'].tolist()))
        _hidden_state_path = utils.checker.get_data_dir() / 'hidden_state.npz'
        utils.checker.save_npz(
            {'selected': np.array([bool(_correct_by_text[c]) for c in CARDS])},
            _hidden_state_path,
        )
        _preview_selected = [bool(v) for v in utils.checker.load_answer(_hidden_state_path)['selected']]
        _hidden_state_path.unlink(missing_ok=True)
        set_selected(_preview_selected)
    # -- end --

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

    # -- start --
    _correct = utils.checker.load_answer(
        utils.checker.get_data_dir() / '.correct_selection.npz'
    )
    utils.cards.save_selection_answer(_correct['cards'], _correct['correct'])
    # -- end --

    widget
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
