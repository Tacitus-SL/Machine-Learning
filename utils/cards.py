import shutil
from pathlib import Path

import numpy as np
import ipywidgets as widgets
from IPython.display import display, Markdown

from . import checker

# These three are the student-visible, task-root files -- static locations,
# used both during grading and when a student runs the task on its own.
INPUTS_PATH = Path('xinputs.npz')
BENCHMARK_PATH = Path('yanswer.npz')
CURRENT_ORDER_PATH = Path('state.npz')
# PUBLIC_INPUTS_PATH, PRIVATE_INPUTS_PATH, and ANSWER_PATH used to be static
# paths here too. They're the checker's own working files, hidden inside a
# temporary directory that's created fresh per run (checker.get_data_dir()),
# so they can no longer be fixed at import time -- see each call site below.

CARD_WIDTH = '650px'
MATCH_COLUMN_WIDTH = '260px'
MATCH_CARD_HEIGHT = '112px'
MATCH_WIDGET_WIDTH = '552px'


def _shuffled(cards, seed):
    order = np.random.RandomState(seed).permutation(len(cards))
    return [cards[i] for i in order]


def _is_valid_npz(path):
    """A plain path.exists() isn't enough to trust a file as already
    generated -- an empty or truncated .npz (e.g. left behind by a tool
    that copies course files without their real content, like the JB
    Academy plugin's Preview snapshot has done) still exists but raises
    EOFError on load. Treat anything that fails to load as absent, so a
    fresh run repairs it instead of crashing forever on stale garbage.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            list(data.files)
        return True
    except Exception:
        return False


def generate_inputs(canonical_cards, fixed_items=None, public_seed=0, private_seed=42):
    """
    Write the public/private shuffles of `canonical_cards` to their npz
    files and record the canonical order in answer_benchmark.npz -- the
    only file the correct order is ever written to, example:
    canonical_cards=['A', 'B', 'C'] -> inputs_public.npz holds CARDS
    shuffled with public_seed, inputs_private.npz holds it shuffled with
    private_seed, answer_benchmark.npz holds reordered_cards=['A','B','C'].

    Parameters
    ----------
    canonical_cards : list
        Cards in their correct order; never written unshuffled to a
        public/private inputs file.
    fixed_items : list or None
        Unshuffled column for matching-style tasks (see
        `build_matching_widget`), stored alongside CARDS in both inputs
        files when given.
    public_seed : int
        RandomState seed for the public shuffle.
    private_seed : int
        RandomState seed for the private shuffle.

    Returns
    -------
    tuple of dict
        `(inputs_contents, answer_contents)` -- the contents of
        inputs.npz and answer_benchmark.npz after this call.
    """
    public_data = {'CARDS': _shuffled(canonical_cards, public_seed)}
    private_data = {'CARDS': _shuffled(canonical_cards, private_seed)}
    if fixed_items is not None:
        public_data['fixed_items'] = fixed_items
        private_data['fixed_items'] = fixed_items

    public_path = checker.get_data_dir() / 'inputs_public.npz'
    private_path = checker.get_data_dir() / 'inputs_private.npz'
    checker.save_npz(public_data, public_path)
    checker.save_npz(private_data, private_path)

    if not _is_valid_npz(INPUTS_PATH):
        shutil.copyfile(public_path, INPUTS_PATH)

    checker.save_npz({'reordered_cards': canonical_cards}, BENCHMARK_PATH)

    inputs_contents = checker.load_answer(INPUTS_PATH)
    answer_contents = checker.load_answer(BENCHMARK_PATH)
    return inputs_contents, answer_contents


def load_cards():
    """Load CARDS from inputs.npz -- already shuffled, never canonical."""
    return list(checker.load_answer(INPUTS_PATH)['CARDS'])


def load_fixed_items():
    """Load the fixed (never shuffled, never movable) column written by
    generate_inputs(..., fixed_items=...)."""
    return list(checker.load_answer(INPUTS_PATH)['fixed_items'])


def _load_or_init_current_order(cards):
    if _is_valid_npz(CURRENT_ORDER_PATH):
        return list(checker.load_answer(CURRENT_ORDER_PATH)['current_order'])
    order = list(np.random.permutation(len(cards)))
    _save_current_order(order)
    return order


def _save_current_order(order):
    checker.save_npz({'current_order': np.array(order)}, CURRENT_ORDER_PATH)


def build_ordering_widget(cards):
    """
    Render `cards` as a reorderable widget, one bordered frame per card
    with its own up/down buttons, and return the live index list backing
    it, example: cards=['A','B'] displays two framed cards and returns
    current_order=[0, 1], mutated in place by the widget's click handlers.

    Parameters
    ----------
    cards : list of str
        Markdown-rendered card texts, in their initial (shuffled) order.

    Returns
    -------
    list of int
        `current_order` -- indices into `cards`; persisted to
        current_order.npz on every click, so a fresh, headless
        re-execution resumes where a student's session left off.

    Examples
    --------
    >>> order = build_ordering_widget(['A', 'B', 'C'])
    >>> order
    [2, 0, 1]
    """
    current_order = _load_or_init_current_order(cards)
    card_outputs = [
        widgets.Output(layout=widgets.Layout(max_width=CARD_WIDTH, flex='1 1 auto'))
        for _ in cards
    ]

    def render():
        for position, idx in enumerate(current_order):
            card_outputs[position].clear_output(wait=True)
            with card_outputs[position]:
                display(Markdown(cards[idx]))

    def make_handler(position, delta):
        def handler(_button):
            neighbor = position + delta
            if 0 <= neighbor < len(current_order):
                current_order[position], current_order[neighbor] = (
                    current_order[neighbor], current_order[position],
                )
                _save_current_order(current_order)
                render()
        return handler

    frames = []
    for position in range(len(cards)):
        up = widgets.Button(description='▲', layout=widgets.Layout(width='36px', margin='0 0 4px 0'))
        down = widgets.Button(description='▼', layout=widgets.Layout(width='36px', margin='0'))
        up.on_click(make_handler(position, -1))
        down.on_click(make_handler(position, 1))
        arrows = widgets.VBox(
            [up, down],
            layout=widgets.Layout(margin='6px 14px 6px 6px', flex='0 0 auto'),
        )
        frame = widgets.HBox(
            [arrows, card_outputs[position]],
            layout=widgets.Layout(
                border='1px solid gray',
                padding='8px',
                margin='4px 0',
                align_items='center',
                max_width=CARD_WIDTH,
            ),
        )
        frames.append(frame)

    display(widgets.VBox(frames, layout=widgets.Layout(max_width=CARD_WIDTH)))
    render()

    return current_order


def build_matching_widget(fixed_items, movable_items):
    """Render an interactive two-column widget: `fixed_items` is
    displayed once, in the given order, and never moves; `movable_items`
    is a reorderable column with its own up/down buttons, driven by a
    per-student stable shuffle/persistence (see `_load_or_init_current_order`).
    Returns the live `current_order` list backing the movable column. No
    position numbers are shown on either column -- the fixed column is
    what anchors each row instead.
    """
    if len(fixed_items) != len(movable_items):
        raise ValueError("fixed_items and movable_items must be the same length")

    current_order = _load_or_init_current_order(movable_items)

    left_outputs = [
        widgets.Output(layout=widgets.Layout(max_width=MATCH_COLUMN_WIDTH, flex='1 1 auto'))
        for _ in fixed_items
    ]
    right_outputs = [
        widgets.Output(layout=widgets.Layout(max_width=MATCH_COLUMN_WIDTH, flex='1 1 auto'))
        for _ in movable_items
    ]

    for position, item in enumerate(fixed_items):
        with left_outputs[position]:
            display(Markdown(item))

    def render():
        for position, idx in enumerate(current_order):
            right_outputs[position].clear_output(wait=True)
            with right_outputs[position]:
                display(Markdown(movable_items[idx]))

    def make_handler(position, delta):
        def handler(_button):
            neighbor = position + delta
            if 0 <= neighbor < len(current_order):
                current_order[position], current_order[neighbor] = (
                    current_order[neighbor], current_order[position],
                )
                _save_current_order(current_order)
                render()
        return handler

    frames = []
    for position in range(len(movable_items)):
        up = widgets.Button(description='▲', layout=widgets.Layout(width='36px', margin='0 0 4px 0'))
        down = widgets.Button(description='▼', layout=widgets.Layout(width='36px', margin='0'))
        up.on_click(make_handler(position, -1))
        down.on_click(make_handler(position, 1))
        arrows = widgets.VBox(
            [up, down],
            layout=widgets.Layout(margin='6px 14px 6px 6px', flex='0 0 auto'),
        )
        separator = widgets.Box(
            layout=widgets.Layout(
                border_left='1px solid gray',
                margin='0 14px',
                align_self='stretch',
                flex='0 0 auto',
            ),
        )
        frame = widgets.HBox(
            [left_outputs[position], separator, arrows, right_outputs[position]],
            layout=widgets.Layout(
                border='1px solid gray',
                padding='8px',
                margin='4px 0',
                align_items='center',
                max_width=CARD_WIDTH,
            ),
        )
        frames.append(frame)

    display(widgets.VBox(frames, layout=widgets.Layout(max_width=CARD_WIDTH)))
    render()

    return current_order


def save_answer(cards, current_order):
    """Save the current reordering as this task's answer.npz."""
    answer = {'reordered_cards': [cards[i] for i in current_order]}
    checker.save_npz(answer, checker.get_data_dir() / 'answer.npz')


def _cards_changed(stored_cards, current_cards):
    """Whether the authored card text differs from what's stored in
    inputs_path -- compared as a sorted list, not the stored order
    (inputs_path holds a SHUFFLED copy, never the original sequence), and
    not a set (a set would miss a duplicate being added or removed).
    """
    return (
        sorted(np.asarray(stored_cards).tolist())
        != sorted(np.asarray(current_cards).tolist())
    )


def generate_ordering_inputs(
        ordered_cards,
        cards_key='cards',
        private_path=None,
        inputs_path=INPUTS_PATH):

    if private_path is None:
        private_path = checker.get_data_dir() / 'inputs_private.npz'

    ordered_cards = np.asarray(ordered_cards)
    stale = False
    if _is_valid_npz(inputs_path):
        stored = checker.load_answer(inputs_path)[cards_key]
        stale = _cards_changed(stored, ordered_cards)

    if _is_valid_npz(inputs_path) and not stale:
        # Already generated on some earlier run with the SAME card text --
        # reuse the existing, committed shuffle rather than drawing a
        # fresh one. inputs_path is the persistent, task-root file;
        # private_path only needs to hold a copy of it for this run.
        shutil.copy(inputs_path, private_path)
    else:
        shuffled = ordered_cards[np.random.permutation(len(ordered_cards))]
        checker.save_npz({cards_key: shuffled}, private_path)
        shutil.copy(private_path, inputs_path)
        checker.refresh_preserved_inputs(inputs_path)
        if stale:
            # The authored cards changed -- any saved reordering
            # (state.npz) refers to the OLD sequence and is meaningless
            # against the new one. Drop it so the widget restarts at a
            # fresh identity order instead of the new cards in a
            # leftover, unrelated order.
            CURRENT_ORDER_PATH.unlink(missing_ok=True)


def load_ordering_cards(cards_key='cards', inputs_path=INPUTS_PATH):
    return checker.load_answer(inputs_path)[cards_key]


def init_ordering_state(mo, cards, order_path=CURRENT_ORDER_PATH):
    if _is_valid_npz(order_path):
        initial_order = [int(i) for i in checker.load_answer(order_path)['current_order']]
    else:
        # cards is already shuffled (generate_ordering_inputs) -- start
        # the widget at identity order, not a second, independent shuffle.
        initial_order = list(range(len(cards)))
    return mo.state(initial_order, allow_self_loops=True)


def build_ordering_widget_marimo(
        mo,
        cards,
        get_order,
        set_order,
        order_path=CURRENT_ORDER_PATH):

    order = get_order()

    def save_order(new_order):
        checker.save_npz({'current_order': np.array(new_order)}, order_path)

    def make_handler(delta, position):
        def handler(_value):
            neighbor = position + delta
            if 0 <= neighbor < len(order):
                new_order = list(order)
                new_order[position], new_order[neighbor] = new_order[neighbor], new_order[position]
                save_order(new_order)
                set_order(new_order)
        return handler

    up_buttons = mo.ui.array([
        mo.ui.button(label="▲", on_click=make_handler(-1, position))
        for position in range(len(order))
    ])
    down_buttons = mo.ui.array([
        mo.ui.button(label="▼", on_click=make_handler(1, position))
        for position in range(len(order))
    ])

    rows = []
    for position, idx in enumerate(order):
        text = mo.md(cards[idx]).style({"text-align": "left", "width": "100%"})
        rows.append(
            mo.hstack(
                [mo.vstack([up_buttons[position], down_buttons[position]]), text],
                align="center", justify="start", gap=1,
            ).style({
                "border": "1px solid gray", "border-radius": "4px",
                "padding": "8px", "margin": "4px 0",
                "width": "100%", "max-width": CARD_WIDTH,
            })
        )

    return (
        order,
        mo.vstack(rows, align='center'),
        up_buttons,
        down_buttons,
    )


def save_ordering_answer(cards, order=None):
    if order is None:
        order = list(range(len(cards)))
    answer = '\n'.join(cards[order])
    checker.save_npz({'answer': answer}, checker.get_data_dir() / 'answer.npz')


def generate_matching_inputs(
        fixed_cards,
        matching_cards,
        fixed_key='fixed_cards',
        matching_key='matching_cards',
        private_path=None,
        inputs_path=INPUTS_PATH):

    if private_path is None:
        private_path = checker.get_data_dir() / 'inputs_private.npz'

    fixed_cards = np.asarray(fixed_cards)
    matching_cards = np.asarray(matching_cards)
    if len(fixed_cards) != len(matching_cards):
        raise ValueError("fixed_cards and matching_cards must have the same length")

    stale = False
    if _is_valid_npz(inputs_path):
        stored = checker.load_answer(inputs_path)
        stale = (
            _cards_changed(stored[fixed_key], fixed_cards)
            or _cards_changed(stored[matching_key], matching_cards)
        )

    if _is_valid_npz(inputs_path) and not stale:
        # Already generated on some earlier run with the SAME card text --
        # reuse the existing, committed shuffle rather than drawing a
        # fresh one. inputs_path is the persistent, task-root file;
        # private_path only needs to hold a copy of it for this run.
        shutil.copy(inputs_path, private_path)
    else:
        shuffled = matching_cards[np.random.permutation(len(matching_cards))]
        checker.save_npz({fixed_key: fixed_cards, matching_key: shuffled}, private_path)
        shutil.copy(private_path, inputs_path)
        checker.refresh_preserved_inputs(inputs_path)
        if stale:
            # The authored cards changed -- any saved reordering
            # (state.npz) refers to the OLD sequence and is meaningless
            # against the new one. Drop it so the widget restarts at a
            # fresh identity order instead of the new cards in a
            # leftover, unrelated order.
            CURRENT_ORDER_PATH.unlink(missing_ok=True)


def load_matching_cards(fixed_key='fixed_cards', matching_key='matching_cards', inputs_path=INPUTS_PATH):
    inputs = checker.load_answer(inputs_path)
    return inputs[fixed_key], inputs[matching_key]


def build_matching_widget_marimo(
        mo,
        fixed_cards,
        matching_cards,
        get_order,
        set_order,
        order_path=CURRENT_ORDER_PATH):

    if len(fixed_cards) != len(matching_cards):
        raise ValueError(
            "fixed_cards and matching_cards must have the same length"
        )

    order = get_order()

    def save_order(new_order):
        checker.save_npz({'current_order': np.array(new_order)}, order_path)

    def make_handler(delta, position):
        def handler(_value):
            neighbor = position + delta
            if 0 <= neighbor < len(order):
                new_order = list(order)
                new_order[position], new_order[neighbor] = new_order[neighbor], new_order[position]
                save_order(new_order)
                set_order(new_order)
        return handler

    up_buttons = mo.ui.array([
        mo.ui.button(label="▲", on_click=make_handler(-1, position))
        for position in range(len(order))
    ])
    down_buttons = mo.ui.array([
        mo.ui.button(label="▼", on_click=make_handler(1, position))
        for position in range(len(order))
    ])

    rows = []
    for position, matching_index in enumerate(order):
        fixed_text = mo.md(fixed_cards[position]).style({
            'align-items': 'center',
            'border': '1px solid gray',
            'border-radius': '4px',
            'box-sizing': 'border-box',
            'display': 'flex',
            'height': MATCH_CARD_HEIGHT,
            'justify-content': 'center',
            'min-width': MATCH_COLUMN_WIDTH,
            'padding': '12px',
            'text-align': 'center',
            'width': MATCH_COLUMN_WIDTH,
        })
        matching_text = mo.md(matching_cards[matching_index]).style({
            'text-align': 'center',
            'width': '100%',
        })
        controls = mo.vstack([
            up_buttons[position],
            down_buttons[position],
        ])
        matching_card = mo.hstack(
            [controls, matching_text],
            align='center',
            justify='start',
            gap=1,
        ).style({
            'border': '1px solid gray',
            'border-radius': '4px',
            'box-sizing': 'border-box',
            'display': 'flex',
            'height': MATCH_CARD_HEIGHT,
            'align-items': 'center',
            'min-width': MATCH_COLUMN_WIDTH,
            'padding': '8px',
            'width': MATCH_COLUMN_WIDTH,
        })
        rows.append(
            mo.hstack(
                [fixed_text, matching_card],
                align='center',
                justify='center',
                gap=2,
                widths=[0, 0],
            ).style({
                'margin': '8px 0',
                'min-width': MATCH_WIDGET_WIDTH,
                'width': MATCH_WIDGET_WIDTH,
            })
        )

    widget = mo.vstack(rows, align='center').style({
        'min-width': MATCH_WIDGET_WIDTH,
        'width': MATCH_WIDGET_WIDTH,
    }).center()
    return order, widget, up_buttons, down_buttons


def save_matching_answer(matching_cards, order=None):
    if order is None:
        order = list(range(len(matching_cards)))
    answer = '\n'.join(matching_cards[order])
    checker.save_npz({'answer': answer}, checker.get_data_dir() / 'answer.npz')


def init_selection_state(mo, cards, state_path=CURRENT_ORDER_PATH):
    """
    Initialize (and resume) a checkbox-selection widget's per-card state.

    Mirrors init_ordering_state(): if `state_path` already holds a saved
    selection, that is resumed; otherwise every card starts unselected.
    Display order itself is controlled separately, by whichever
    generate_ordering_inputs()/load_ordering_cards() call the caller used
    to produce `cards` (see build_selection_widget_marimo()) -- reused
    as-is here, since shuffling display order and detecting stale card
    text is identical work whether the widget reorders cards or selects
    them.

    Parameters
    ----------
    mo : module
        The imported `marimo` module (passed in, not imported here, so
        this stays usable outside a marimo notebook cell).
    cards : list or np.ndarray
        Card texts, in their display order.
    state_path : Path
        Where the live selection is persisted between runs.

    Returns
    -------
    tuple
        `(get_selected, set_selected)`, a marimo state pair holding a
        list of bool, one per card.
    """
    if _is_valid_npz(state_path):
        selected = [bool(v) for v in checker.load_answer(state_path)['selected']]
    else:
        selected = [False] * len(cards)
    return mo.state(selected, allow_self_loops=True)


def build_selection_widget_marimo(
        mo,
        cards,
        get_selected,
        set_selected,
        state_path=CURRENT_ORDER_PATH):
    """
    Render `cards` as a column of checkboxes, one per card, in place of
    the up/down reordering controls used by build_ordering_widget_marimo().
    Checking a card marks it as one of the student's selected statements;
    display order never changes. Selection state is persisted to
    `state_path` on every click, the same way build_ordering_widget_marimo()
    persists its order -- a fresh, headless re-execution resumes where a
    student's session left off.

    Parameters
    ----------
    mo : module
        The imported `marimo` module.
    cards : list or np.ndarray
        Card texts, in their (already shuffled) display order.
    get_selected, set_selected : callables
        The marimo state pair returned by init_selection_state().
    state_path : Path
        Where the live selection is persisted between runs.

    Returns
    -------
    tuple
        `(selected, widget, checkboxes)` -- `selected` is the live list
        of bool backing the widget; `widget` is the marimo layout to
        display; `checkboxes` is the `mo.ui.array` of checkbox elements
        (referencing it in another cell keeps their callbacks registered,
        the same way the ordering widget's up/down button arrays do).
    """
    selected = get_selected()

    def save_selected(new_selected):
        checker.save_npz({'selected': np.array(new_selected)}, state_path)

    def make_handler(position):
        def handler(value):
            new_selected = list(selected)
            new_selected[position] = bool(value)
            save_selected(new_selected)
            set_selected(new_selected)
        return handler

    checkboxes = mo.ui.array([
        mo.ui.checkbox(value=selected[position], on_change=make_handler(position))
        for position in range(len(cards))
    ])

    rows = []
    for position, card in enumerate(cards):
        text = mo.md(card).style({"text-align": "left", "width": "100%"})
        rows.append(
            mo.hstack(
                [checkboxes[position], text],
                align="center", justify="start", gap=1,
            ).style({
                "border": "1px solid gray", "border-radius": "4px",
                "padding": "8px", "margin": "4px 0",
                "width": "100%", "max-width": CARD_WIDTH,
            })
        )

    return (
        selected,
        mo.vstack(rows, align='center'),
        checkboxes,
    )


def save_selection_answer(cards, selected=None):
    """
    Save the currently checked cards as this task's answer.npz.

    The saved value is order-independent (sorted): which cards are
    checked is what is graded, not the order they were checked in or the
    order they are displayed in.

    Parameters
    ----------
    cards : list or np.ndarray
        Card texts, aligned element-for-element with `selected`.
    selected : list of bool or None
        Which cards are checked. `None` (e.g. a blank/never-touched
        widget) is treated as nothing selected.
    """
    if selected is None:
        selected = [False] * len(cards)
    cards = np.asarray(cards)
    mask = np.asarray(selected, dtype=bool)
    answer = {'selected_cards': sorted(cards[mask].tolist())}
    checker.save_npz(answer, checker.get_data_dir() / 'answer.npz')
