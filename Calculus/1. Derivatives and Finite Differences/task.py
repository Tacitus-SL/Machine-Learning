import marimo

__generated_with = "0.24.0"
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

    import json

    import numpy as np

    DX = 0.1  # forward-difference step: dx/dy are between x and x+DX

    # Each entry: internal key, display label for f, f, derivative's display
    # label, f', the function's own valid plotting/slider domain, and the
    # target x this function is graded against -- the slider's own position
    # is never checked (it can sit anywhere), but the entered dy/dx is
    # always compared to the derivative AT THIS x specifically, computed
    # fresh regardless of where the slider currently is (see the
    # achieved-flag logic in each build cell). Target x = 1.0 everywhere
    # it's valid (checked against each function's domain, with DX margin);
    # arcsin is the one exception -- 1.0 +/- DX would leave its [-1, 1]
    # domain, so it uses 0.5 instead.
    FUNCTIONS = [
        ("square", "x²", lambda x: x ** 2, "2x", lambda x: 2 * x, (-5.0, 5.0), 1.0),
        ("cube", "x³", lambda x: x ** 3, "3x²", lambda x: 3 * x ** 2, (-5.0, 5.0), 1.0),
        ("sqrt", "√x", np.sqrt, "1/(2√x)", lambda x: 1 / (2 * np.sqrt(x)), (0.1, 5.0), 1.0),
        ("sin", "sin x", np.sin, "cos x", np.cos, (-5.0, 5.0), 1.0),
        ("tan", "tan x", np.tan, "sec²x", lambda x: 1 / np.cos(x) ** 2, (-1.2, 1.2), 1.0),
        ("exp", "eˣ", np.exp, "eˣ", np.exp, (-5.0, 5.0), 1.0),
        ("log", "log x", np.log, "1/x", lambda x: 1 / x, (0.15, 5.0), 1.0),
        ("arcsin", "arcsin x", np.arcsin, "1/√(1-x²)", lambda x: 1 / np.sqrt(1 - x ** 2), (-0.9, 0.9), 0.5),
    ]

    # Dropdown options are keyed by the derivative's own label (what the
    # student picks among); tab titles use the function's label instead
    # (what's actually being explored in that tab) -- these are two
    # different lookups and must not be confused with each other.
    OPTIONS = {deriv_label: key for key, _, _, deriv_label, _, _, _ in FUNCTIONS}
    LABEL_BY_KEY = {key: deriv_label for key, _, _, deriv_label, _, _, _ in FUNCTIONS}
    FUNCTION_LABEL_BY_KEY = {key: label for key, label, _, _, _, _, _ in FUNCTIONS}

    # 5 points per function, easy to reach with the slider's 0.05 step,
    # spread across each function's own domain (not one shared set --
    # e.g. arcsin's domain excludes all of -2..2 but 0). Matches task.md's
    # "good spread to aim for" list exactly. Ungraded -- purely a by-hand
    # scratchpad, plotted below to compare against the slider-computed
    # trace, not checked against the true derivative anywhere.
    TABLE_POINTS_BY_KEY = {
        "square": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "cube": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "sqrt": [0.5, 1.0, 2.0, 3.0, 4.0],
        "sin": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "tan": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "exp": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "log": [0.5, 1.0, 2.0, 3.0, 4.0],
        "arcsin": [-0.5, -0.25, 0.0, 0.25, 0.5],
    }

    # Fixed seed -- shuffled once, the same order every session, rather
    # than truly random (which would make screenshots/bug reports of "the
    # third option" not reproducible between runs).
    import random

    _option_items = list(OPTIONS.items())
    random.Random(42).shuffle(_option_items)
    OPTIONS = dict(_option_items)

    _state_path = "state.npz"
    if utils.checker.is_valid_npz(_state_path):
        try:
            SAVED_STATE = json.loads(str(utils.checker.load_answer(_state_path)["state"]))
        except Exception:
            SAVED_STATE = {}
    else:
        SAVED_STATE = {}
    return (
        DX,
        FUNCTIONS,
        FUNCTION_LABEL_BY_KEY,
        LABEL_BY_KEY,
        OPTIONS,
        SAVED_STATE,
        TABLE_POINTS_BY_KEY,
        json,
        mo,
        np,
        utils,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Derivatives and Finite Differences

    Eight functions, each in its own tab, accessible in any order at any time.
    For each one:

    1. Start at `x = target`, where the slider begins (its own position isn't
       graded).
    2. Fill in `dy/dx` at every point in the table under the plot — all 5
       points are graded independently.
    3. Move the slider around too, if you want to see more points than the
       table's 5 — each one you visit gets plotted on the main plot as well.
    4. Select the function that fits the plot as close as possible (and is
       probably the derivative).

    There's no live right/wrong feedback — only the actual submission is
    graded. The slider's own position is never checked — what's graded is
    whether `dy/dx` was ever entered correctly at each of the table's 5
    points, regardless of where the slider ends up afterward.

    Each tab is labeled with its internal key (e.g. `square`) so a submission
    report entry like `square_deriv_value_0` (the table's leftmost point)
    through `square_deriv_value_4` (rightmost) can be matched back to the
    right tab and column. The report is pass/fail only — it doesn't reveal
    the correct values.
    """)
    return


@app.cell(hide_code=True)
def _(DX, mo, np):
    # Shared by every per-function build cell below, so the ~60 lines of
    # matplotlib code exist exactly once despite there being 8 independent
    # copies of the widgets that feed it. Each function needs its own
    # separately-named slider/guess/candidate (not one shared container) so
    # that moving one function's slider only reruns *that* function's own
    # build cell -- not all 8. A mo.ui.dictionary would still batch all 8
    # together under one combined .value, rebuilding every plot on any
    # single change.
    def _safe_curve(func, domain, samples=400):
        _xs = np.linspace(max(domain[0], -5.0), min(domain[1], 5.0), samples)
        with np.errstate(all="ignore"):
            _ys = func(_xs)
        _ys = np.where(np.isfinite(_ys), _ys, np.nan)
        _ys = np.where(np.abs(_ys) <= 5.5, _ys, np.nan)
        return _xs, _ys

    def slider_start_value(entry, saved_state):
        _domain_min, _domain_max = entry[5]
        _step = 0.05
        _saved_x = saved_state.get(entry[0], {}).get("x")
        if isinstance(_saved_x, (int, float)) and _domain_min <= _saved_x <= _domain_max:
            return round(float(_saved_x), 4)
        _steps_to_mid = round((_domain_max - _domain_min) / 2 / _step)
        return round(_domain_min + _steps_to_mid * _step, 4)

    def build_function_plot(
        entry, all_entries, x0, picked_key, trace_xs, trace_ys, table_xs, table_ys,
    ):
        import io

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        _dpi = 100

        def _fixed_size_image(fig):
            # mo.hstack/flex can otherwise stretch or shrink an embedded
            # figure to fill whatever space is available; rendering to a
            # buffer and handing mo.image explicit pixel dimensions (no
            # bbox_inches="tight", so the buffer's own size always matches
            # figsize * dpi exactly) keeps the displayed size fixed
            # regardless of the surrounding layout.
            _buf = io.BytesIO()
            fig.savefig(_buf, format="png", dpi=_dpi)
            _buf.seek(0)
            _width_px = round(fig.get_figwidth() * _dpi)
            _height_px = round(fig.get_figheight() * _dpi)
            plt.close(fig)
            return mo.image(src=_buf, width=_width_px, height=_height_px)

        _key, _label, _f, _deriv_label, _deriv_f, _domain, _target_x0 = entry
        _y0 = float(_f(x0))
        _yplus = float(_f(x0 + DX))
        _fwd_slope = (_yplus - _y0) / DX

        # Window centered on (x, f(x)): +/- 0.2 on x, and on y at least
        # +/- 0.6 -- widened when the curve is steep so that the point
        # (x + DX, f(x + DX)) and its label always fit.
        _zoom_xmin, _zoom_xmax = x0 - 0.2, x0 + 0.2
        _zoom_half_height = max(0.6, 1.5 * abs(_yplus - _y0))
        _zoom_ymin, _zoom_ymax = _y0 - _zoom_half_height, _y0 + _zoom_half_height

        _fig_main, _ax_main = plt.subplots(figsize=(5.5, 5))
        _xs, _ys = _safe_curve(_f, _domain)
        _ax_main.plot(_xs, _ys, color="#7c3aed", linewidth=2.5, label=_label)

        # Every x visited by the slider gets its own finite-difference
        # dy/dx plotted here -- traces out the derivative empirically as
        # the student explores, independent of whether any answer is
        # actually correct (there's no live correctness signal anymore).
        if trace_xs:
            _ax_main.scatter(
                trace_xs, trace_ys, color="#2563eb", s=25, zorder=4.5,
                label="dy/dx samples",
            )

        # By-hand table estimates: 5 fixed x's, student-entered dy/dx.
        # Defaults to 0.0 until filled in (same as the main guess field
        # above), so an untouched table shows a flat row at y=0 rather
        # than nothing -- consistent with how the rest of this task
        # already treats an untouched numeric input as a literal 0
        # answer, not a distinct "unanswered" state.
        _ax_main.scatter(
            table_xs, table_ys, color="#16a34a", marker="x", s=60,
            linewidth=2, zorder=4.8, label="your table estimate",
        )

        # Only the candidate the student picked is ever plotted -- never
        # the true derivative curve, so there's nothing to visually
        # compare against except the shape of f itself.
        if picked_key is not None:
            _cand_entry = next(e for e in all_entries if e[0] == picked_key)
            _cand_deriv_label, _cand_deriv_f, _cand_domain = (
                _cand_entry[3], _cand_entry[4], _cand_entry[5],
            )
            _cxs, _cys = _safe_curve(_cand_deriv_f, _cand_domain)
            _ax_main.plot(
                _cxs, _cys, color="#f59e0b", linewidth=2, linestyle=":",
                label=f"your guess: {_cand_deriv_label}",
            )

        # Two-point line through (x, f(x)) and (x+0.1, f(x+0.1)), extended
        # across the whole plot so it reads as a tangent line.
        _ax_main.plot(
            [-5, 5], [_y0 + _fwd_slope * (-5 - x0), _y0 + _fwd_slope * (5 - x0)],
            "--", color="#94a3b8", linewidth=1.5, zorder=4,
        )
        _ax_main.add_patch(
            Rectangle(
                (_zoom_xmin, max(_zoom_ymin, -5.0)),
                _zoom_xmax - _zoom_xmin,
                min(_zoom_ymax, 5.0) - max(_zoom_ymin, -5.0),
                fill=False, edgecolor="#94a3b8", linestyle="--", linewidth=1.2, zorder=3,
            )
        )

        _ax_main.scatter([x0], [_y0], color="#f97316", zorder=5)
        _ax_main.set_xlim(-5, 5)
        _ax_main.set_ylim(-5, 5)
        _ax_main.set_xlabel("x")
        _ax_main.set_ylabel("y")
        _ax_main.legend(loc="upper left", fontsize=8)
        _ax_main.grid(True, linewidth=0.5, alpha=0.5)
        _ax_main.set_title(f"f(x) = {_label}")
        # Explicit margins instead of tight_layout(): reclaims the extra
        # whitespace tight_layout() tends to leave, so the grid/plot area
        # fills more of the fixed figure size instead of shrinking to
        # leave a large border around the axis labels.
        _fig_main.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.09)

        _fig_zoom, _ax_zoom = plt.subplots(figsize=(3.2, 5))
        _zxs, _zys = _safe_curve(_f, _domain)
        _ax_zoom.plot(_zxs, _zys, color="#7c3aed", linewidth=2, zorder=1)

        _ax_zoom.plot(
            [_zoom_xmin, _zoom_xmax],
            [_y0 + _fwd_slope * (_zoom_xmin - x0), _y0 + _fwd_slope * (_zoom_xmax - x0)],
            "--", color="#94a3b8", linewidth=1.5, zorder=2,
        )

        # Cathetes of the right triangle formed by x -> x+0.1.
        _ax_zoom.plot(
            [x0, x0 + DX], [_y0, _y0],
            "--", color="#f59e0b", linewidth=1.5, zorder=3,
        )
        _ax_zoom.plot(
            [x0 + DX, x0 + DX], [_y0, _yplus],
            "--", color="#f59e0b", linewidth=1.5, zorder=3,
        )
        # dx/dy live in one fixed corner box (axes-fraction coordinates, not
        # data coordinates) rather than floating near the legs -- that way
        # they never drift into the point labels below regardless of where
        # x0 sits or how steep the local slope is.
        _ax_zoom.text(
            0.03, 0.97, f"dx = {DX:.2f}\ndy = {_yplus - _y0:.6f}",
            transform=_ax_zoom.transAxes, ha="left", va="top", fontsize=8,
            color="#b45309",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#f59e0b", alpha=0.9),
        )

        _ax_zoom.scatter([x0 + DX], [_yplus], color="#7c3aed", s=60, zorder=5)
        _ax_zoom.scatter([x0], [_y0], color="#f97316", s=70, zorder=6)
        # Opposite corners (upper-left / lower-right) so the two point
        # labels stay apart regardless of whether the secant slopes up or
        # down -- boxed for readability over the curve and grid.
        _ax_zoom.annotate(
            f"x={x0:.2f}\ny={_y0:.6f}", (x0, _y0),
            textcoords="offset points", xytext=(-8, 10), ha="right", va="bottom", fontsize=8,
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )
        _ax_zoom.annotate(
            f"x+0.1={x0 + DX:.2f}\ny={_yplus:.6f}", (x0 + DX, _yplus),
            textcoords="offset points", xytext=(8, -10), ha="left", va="top", fontsize=8,
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )
        _ax_zoom.set_xlim(_zoom_xmin, _zoom_xmax)
        _ax_zoom.set_ylim(_zoom_ymin, _zoom_ymax)
        _ax_zoom.set_xlabel("x")
        _ax_zoom.set_title("zoomed: x ± 0.2, y ± 0.6")
        _ax_zoom.grid(True, linewidth=0.5, alpha=0.5)
        _fig_zoom.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.09)

        # No `widths=` here: mo.hstack turns widths into a CSS "flex: N"
        # on each child, whose "flex-basis: 0%" discards the image's own
        # intrinsic size and grows it to fill its share of the container
        # instead -- which is exactly why the rendered width was tracking
        # the page width regardless of the explicit size passed to
        # mo.image above. Without it, each child keeps its natural size.
        return mo.hstack(
            [_fixed_size_image(_fig_main), _fixed_size_image(_fig_zoom)],
            justify="start", gap=0,
        )

    return build_function_plot, slider_start_value


@app.cell(hide_code=True)
def _(FUNCTIONS, LABEL_BY_KEY, OPTIONS, SAVED_STATE, mo, slider_start_value):
    _entry = next(e for e in FUNCTIONS if e[0] == "square")
    _saved = SAVED_STATE.get("square", {})
    _saved_table = _saved.get("table")
    if not isinstance(_saved_table, list) or len(_saved_table) != 5:
        _saved_table = [None] * 5
    square_slider = mo.ui.slider(
        start=_entry[5][0], stop=_entry[5][1], step=0.05,
        value=slider_start_value(_entry, SAVED_STATE),
        label="x (slider position isn't graded -- fill in the table below)",
        full_width=True, show_value=True,
    )
    square_candidate = mo.ui.dropdown(
        options=OPTIONS, value=LABEL_BY_KEY.get(_saved.get("candidate")),
        label="Which is the derivative?", searchable=True,
    )
    # Table cells: 5 independently-named widgets, not a list/dict -- same
    # reason the slider/candidate above are separately named, a shared
    # container wouldn't give per-widget reactivity. Bare (label="") --
    # the build cell draws each point's x above its own cell instead, so
    # this reads as a real two-row table (x row, dy/dx row) rather than 5
    # individually-captioned boxes. Every point is graded independently
    # (not just one target column).
    square_table_0 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[0]) if isinstance(_saved_table[0], (int, float)) else 0.0,
        label="",
    )
    square_table_1 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[1]) if isinstance(_saved_table[1], (int, float)) else 0.0,
        label="",
    )
    square_table_2 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[2]) if isinstance(_saved_table[2], (int, float)) else 0.0,
        label="",
    )
    square_table_3 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[3]) if isinstance(_saved_table[3], (int, float)) else 0.0,
        label="",
    )
    square_table_4 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[4]) if isinstance(_saved_table[4], (int, float)) else 0.0,
        label="",
    )
    return (
        square_candidate,
        square_slider,
        square_table_0,
        square_table_1,
        square_table_2,
        square_table_3,
        square_table_4,
    )


@app.cell(hide_code=True)
def _(mo):
    get_square_trace, set_square_trace = mo.state({"xs": [], "ys": []})
    return get_square_trace, set_square_trace


@app.cell(hide_code=True)
def _(
    DX,
    FUNCTIONS,
    TABLE_POINTS_BY_KEY,
    build_function_plot,
    get_square_trace,
    mo,
    set_square_trace,
    square_candidate,
    square_slider,
    square_table_0,
    square_table_1,
    square_table_2,
    square_table_3,
    square_table_4,
):
    _entry = next(e for e in FUNCTIONS if e[0] == "square")
    square_x = float(square_slider.value)
    square_candidate_value = square_candidate.value

    _f = _entry[2]
    _true_deriv = (_f(square_x + DX) - _f(square_x)) / DX
    _trace = get_square_trace()
    if square_x not in _trace["xs"]:
        _trace = {"xs": _trace["xs"] + [square_x], "ys": _trace["ys"] + [_true_deriv]}
        set_square_trace(_trace)

    square_table_values = [
        float(square_table_0.value or 0.0), float(square_table_1.value or 0.0), float(square_table_2.value or 0.0),
        float(square_table_3.value or 0.0), float(square_table_4.value or 0.0),
    ]
    # Every table point is graded independently -- not just one target x
    # (the slider's position is never checked either). Each point's true
    # derivative is computed fresh from its OWN x, never from the
    # slider's current position (_true_deriv above is separate, only
    # used for the live trace plot).
    #
    # bool(...) wrapper: entries of square_point_true_derivs are
    # numpy.float64 for every function using a numpy ufunc (sqrt, sin,
    # tan, exp, log, arcsin), which makes the comparison a numpy.bool_
    # rather than a plain bool -- and numpy.bool_ isn't JSON-serializable.
    #
    # Computed fresh every time from the table's CURRENT values -- no
    # persisted "once true, stays true" flags. A cell already keeps
    # whatever the student last typed regardless of slider movement, so
    # there's nothing to remember beyond that; a persisted flag would
    # (and did) let a since-overwritten wrong guess still read as correct
    # because it had been right at some earlier point.
    square_point_true_derivs = [
        (_f(_p + DX) - _f(_p)) / DX for _p in TABLE_POINTS_BY_KEY["square"]
    ]
    square_point_achieved = [
        bool(abs(square_table_values[_i] - square_point_true_derivs[_i]) < 0.01)
        for _i in range(5)
    ]

    _plots = build_function_plot(
        _entry, FUNCTIONS, square_x, square_candidate_value, _trace["xs"], _trace["ys"],
        TABLE_POINTS_BY_KEY["square"], square_table_values,
    )
    # Shown so a report entry like "square_deriv_value_0" can be traced
    # back to this tab.
    _key_label = mo.md("`square`")
    _x_label = mo.md(f"**x = {square_x:.4g}**")
    _table_points = TABLE_POINTS_BY_KEY["square"]
    _table_widgets = [
        square_table_0, square_table_1, square_table_2, square_table_3, square_table_4,
    ]
    _table_columns = [mo.vstack([mo.md("**x**"), mo.md("**dy/dx**")])]
    for _point, _widget in zip(_table_points, _table_widgets):
        _table_columns.append(mo.vstack([mo.md(f"**{_point:g}**"), _widget]))
    _table_row = mo.hstack(_table_columns, justify="start", gap=1)
    square_tab = mo.vstack(
        [
            _key_label, _x_label, square_slider, _plots,
            mo.md("Fill in `dy/dx` at every point in the table -- all 5 are graded:"),
            _table_row, square_candidate,
        ],
        gap=1,
    )
    return (
        square_candidate_value,
        square_point_achieved,
        square_point_true_derivs,
        square_tab,
        square_table_values,
        square_x,
    )


@app.cell(hide_code=True)
def _(FUNCTIONS, LABEL_BY_KEY, OPTIONS, SAVED_STATE, mo, slider_start_value):
    _entry = next(e for e in FUNCTIONS if e[0] == "cube")
    _saved = SAVED_STATE.get("cube", {})
    _saved_table = _saved.get("table")
    if not isinstance(_saved_table, list) or len(_saved_table) != 5:
        _saved_table = [None] * 5
    cube_slider = mo.ui.slider(
        start=_entry[5][0], stop=_entry[5][1], step=0.05,
        value=slider_start_value(_entry, SAVED_STATE),
        label="x (slider position isn't graded -- fill in the table below)",
        full_width=True, show_value=True,
    )
    cube_candidate = mo.ui.dropdown(
        options=OPTIONS, value=LABEL_BY_KEY.get(_saved.get("candidate")),
        label="Which is the derivative?", searchable=True,
    )
    # Table cells: 5 independently-named widgets, not a list/dict -- same
    # reason the slider/candidate above are separately named, a shared
    # container wouldn't give per-widget reactivity. Bare (label="") --
    # the build cell draws each point's x above its own cell instead, so
    # this reads as a real two-row table (x row, dy/dx row) rather than 5
    # individually-captioned boxes. Every point is graded independently
    # (not just one target column).
    cube_table_0 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[0]) if isinstance(_saved_table[0], (int, float)) else 0.0,
        label="",
    )
    cube_table_1 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[1]) if isinstance(_saved_table[1], (int, float)) else 0.0,
        label="",
    )
    cube_table_2 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[2]) if isinstance(_saved_table[2], (int, float)) else 0.0,
        label="",
    )
    cube_table_3 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[3]) if isinstance(_saved_table[3], (int, float)) else 0.0,
        label="",
    )
    cube_table_4 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[4]) if isinstance(_saved_table[4], (int, float)) else 0.0,
        label="",
    )
    return (
        cube_candidate,
        cube_slider,
        cube_table_0,
        cube_table_1,
        cube_table_2,
        cube_table_3,
        cube_table_4,
    )


@app.cell(hide_code=True)
def _(mo):
    get_cube_trace, set_cube_trace = mo.state({"xs": [], "ys": []})
    return get_cube_trace, set_cube_trace


@app.cell(hide_code=True)
def _(
    DX,
    FUNCTIONS,
    TABLE_POINTS_BY_KEY,
    build_function_plot,
    cube_candidate,
    cube_slider,
    cube_table_0,
    cube_table_1,
    cube_table_2,
    cube_table_3,
    cube_table_4,
    get_cube_trace,
    mo,
    set_cube_trace,
):
    _entry = next(e for e in FUNCTIONS if e[0] == "cube")
    cube_x = float(cube_slider.value)
    cube_candidate_value = cube_candidate.value

    _f = _entry[2]
    _true_deriv = (_f(cube_x + DX) - _f(cube_x)) / DX
    _trace = get_cube_trace()
    if cube_x not in _trace["xs"]:
        _trace = {"xs": _trace["xs"] + [cube_x], "ys": _trace["ys"] + [_true_deriv]}
        set_cube_trace(_trace)

    cube_table_values = [
        float(cube_table_0.value or 0.0), float(cube_table_1.value or 0.0), float(cube_table_2.value or 0.0),
        float(cube_table_3.value or 0.0), float(cube_table_4.value or 0.0),
    ]
    # Every table point is graded independently -- not just one target x
    # (the slider's position is never checked either). Each point's true
    # derivative is computed fresh from its OWN x, never from the
    # slider's current position (_true_deriv above is separate, only
    # used for the live trace plot).
    #
    # bool(...) wrapper: entries of cube_point_true_derivs are
    # numpy.float64 for every function using a numpy ufunc (sqrt, sin,
    # tan, exp, log, arcsin), which makes the comparison a numpy.bool_
    # rather than a plain bool -- and numpy.bool_ isn't JSON-serializable.
    #
    # Computed fresh every time from the table's CURRENT values -- no
    # persisted "once true, stays true" flags. A cell already keeps
    # whatever the student last typed regardless of slider movement, so
    # there's nothing to remember beyond that; a persisted flag would
    # (and did) let a since-overwritten wrong guess still read as correct
    # because it had been right at some earlier point.
    cube_point_true_derivs = [
        (_f(_p + DX) - _f(_p)) / DX for _p in TABLE_POINTS_BY_KEY["cube"]
    ]
    cube_point_achieved = [
        bool(abs(cube_table_values[_i] - cube_point_true_derivs[_i]) < 0.01)
        for _i in range(5)
    ]

    _plots = build_function_plot(
        _entry, FUNCTIONS, cube_x, cube_candidate_value, _trace["xs"], _trace["ys"],
        TABLE_POINTS_BY_KEY["cube"], cube_table_values,
    )
    # Shown so a report entry like "cube_deriv_value_0" can be traced
    # back to this tab.
    _key_label = mo.md("`cube`")
    _x_label = mo.md(f"**x = {cube_x:.4g}**")
    _table_points = TABLE_POINTS_BY_KEY["cube"]
    _table_widgets = [
        cube_table_0, cube_table_1, cube_table_2, cube_table_3, cube_table_4,
    ]
    _table_columns = [mo.vstack([mo.md("**x**"), mo.md("**dy/dx**")])]
    for _point, _widget in zip(_table_points, _table_widgets):
        _table_columns.append(mo.vstack([mo.md(f"**{_point:g}**"), _widget]))
    _table_row = mo.hstack(_table_columns, justify="start", gap=1)
    cube_tab = mo.vstack(
        [
            _key_label, _x_label, cube_slider, _plots,
            mo.md("Fill in `dy/dx` at every point in the table -- all 5 are graded:"),
            _table_row, cube_candidate,
        ],
        gap=1,
    )
    return (
        cube_candidate_value,
        cube_point_achieved,
        cube_point_true_derivs,
        cube_tab,
        cube_table_values,
        cube_x,
    )


@app.cell(hide_code=True)
def _(FUNCTIONS, LABEL_BY_KEY, OPTIONS, SAVED_STATE, mo, slider_start_value):
    _entry = next(e for e in FUNCTIONS if e[0] == "sqrt")
    _saved = SAVED_STATE.get("sqrt", {})
    _saved_table = _saved.get("table")
    if not isinstance(_saved_table, list) or len(_saved_table) != 5:
        _saved_table = [None] * 5
    sqrt_slider = mo.ui.slider(
        start=_entry[5][0], stop=_entry[5][1], step=0.05,
        value=slider_start_value(_entry, SAVED_STATE),
        label="x (slider position isn't graded -- fill in the table below)",
        full_width=True, show_value=True,
    )
    sqrt_candidate = mo.ui.dropdown(
        options=OPTIONS, value=LABEL_BY_KEY.get(_saved.get("candidate")),
        label="Which is the derivative?", searchable=True,
    )
    # Table cells: 5 independently-named widgets, not a list/dict -- same
    # reason the slider/candidate above are separately named, a shared
    # container wouldn't give per-widget reactivity. Bare (label="") --
    # the build cell draws each point's x above its own cell instead, so
    # this reads as a real two-row table (x row, dy/dx row) rather than 5
    # individually-captioned boxes. Every point is graded independently
    # (not just one target column).
    sqrt_table_0 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[0]) if isinstance(_saved_table[0], (int, float)) else 0.0,
        label="",
    )
    sqrt_table_1 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[1]) if isinstance(_saved_table[1], (int, float)) else 0.0,
        label="",
    )
    sqrt_table_2 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[2]) if isinstance(_saved_table[2], (int, float)) else 0.0,
        label="",
    )
    sqrt_table_3 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[3]) if isinstance(_saved_table[3], (int, float)) else 0.0,
        label="",
    )
    sqrt_table_4 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[4]) if isinstance(_saved_table[4], (int, float)) else 0.0,
        label="",
    )
    return (
        sqrt_candidate,
        sqrt_slider,
        sqrt_table_0,
        sqrt_table_1,
        sqrt_table_2,
        sqrt_table_3,
        sqrt_table_4,
    )


@app.cell(hide_code=True)
def _(mo):
    get_sqrt_trace, set_sqrt_trace = mo.state({"xs": [], "ys": []})
    return get_sqrt_trace, set_sqrt_trace


@app.cell(hide_code=True)
def _(
    DX,
    FUNCTIONS,
    TABLE_POINTS_BY_KEY,
    build_function_plot,
    get_sqrt_trace,
    mo,
    set_sqrt_trace,
    sqrt_candidate,
    sqrt_slider,
    sqrt_table_0,
    sqrt_table_1,
    sqrt_table_2,
    sqrt_table_3,
    sqrt_table_4,
):
    _entry = next(e for e in FUNCTIONS if e[0] == "sqrt")
    sqrt_x = float(sqrt_slider.value)
    sqrt_candidate_value = sqrt_candidate.value

    _f = _entry[2]
    _true_deriv = (_f(sqrt_x + DX) - _f(sqrt_x)) / DX
    _trace = get_sqrt_trace()
    if sqrt_x not in _trace["xs"]:
        _trace = {"xs": _trace["xs"] + [sqrt_x], "ys": _trace["ys"] + [_true_deriv]}
        set_sqrt_trace(_trace)

    sqrt_table_values = [
        float(sqrt_table_0.value or 0.0), float(sqrt_table_1.value or 0.0), float(sqrt_table_2.value or 0.0),
        float(sqrt_table_3.value or 0.0), float(sqrt_table_4.value or 0.0),
    ]
    # Every table point is graded independently -- not just one target x
    # (the slider's position is never checked either). Each point's true
    # derivative is computed fresh from its OWN x, never from the
    # slider's current position (_true_deriv above is separate, only
    # used for the live trace plot).
    #
    # bool(...) wrapper: entries of sqrt_point_true_derivs are
    # numpy.float64 for every function using a numpy ufunc (sqrt, sin,
    # tan, exp, log, arcsin), which makes the comparison a numpy.bool_
    # rather than a plain bool -- and numpy.bool_ isn't JSON-serializable.
    #
    # Computed fresh every time from the table's CURRENT values -- no
    # persisted "once true, stays true" flags. A cell already keeps
    # whatever the student last typed regardless of slider movement, so
    # there's nothing to remember beyond that; a persisted flag would
    # (and did) let a since-overwritten wrong guess still read as correct
    # because it had been right at some earlier point.
    sqrt_point_true_derivs = [
        (_f(_p + DX) - _f(_p)) / DX for _p in TABLE_POINTS_BY_KEY["sqrt"]
    ]
    sqrt_point_achieved = [
        bool(abs(sqrt_table_values[_i] - sqrt_point_true_derivs[_i]) < 0.01)
        for _i in range(5)
    ]

    _plots = build_function_plot(
        _entry, FUNCTIONS, sqrt_x, sqrt_candidate_value, _trace["xs"], _trace["ys"],
        TABLE_POINTS_BY_KEY["sqrt"], sqrt_table_values,
    )
    # Shown so a report entry like "sqrt_deriv_value_0" can be traced
    # back to this tab.
    _key_label = mo.md("`sqrt`")
    _x_label = mo.md(f"**x = {sqrt_x:.4g}**")
    _table_points = TABLE_POINTS_BY_KEY["sqrt"]
    _table_widgets = [
        sqrt_table_0, sqrt_table_1, sqrt_table_2, sqrt_table_3, sqrt_table_4,
    ]
    _table_columns = [mo.vstack([mo.md("**x**"), mo.md("**dy/dx**")])]
    for _point, _widget in zip(_table_points, _table_widgets):
        _table_columns.append(mo.vstack([mo.md(f"**{_point:g}**"), _widget]))
    _table_row = mo.hstack(_table_columns, justify="start", gap=1)
    sqrt_tab = mo.vstack(
        [
            _key_label, _x_label, sqrt_slider, _plots,
            mo.md("Fill in `dy/dx` at every point in the table -- all 5 are graded:"),
            _table_row, sqrt_candidate,
        ],
        gap=1,
    )
    return (
        sqrt_candidate_value,
        sqrt_point_achieved,
        sqrt_point_true_derivs,
        sqrt_tab,
        sqrt_table_values,
        sqrt_x,
    )


@app.cell(hide_code=True)
def _(FUNCTIONS, LABEL_BY_KEY, OPTIONS, SAVED_STATE, mo, slider_start_value):
    _entry = next(e for e in FUNCTIONS if e[0] == "sin")
    _saved = SAVED_STATE.get("sin", {})
    _saved_table = _saved.get("table")
    if not isinstance(_saved_table, list) or len(_saved_table) != 5:
        _saved_table = [None] * 5
    sin_slider = mo.ui.slider(
        start=_entry[5][0], stop=_entry[5][1], step=0.05,
        value=slider_start_value(_entry, SAVED_STATE),
        label="x (slider position isn't graded -- fill in the table below)",
        full_width=True, show_value=True,
    )
    sin_candidate = mo.ui.dropdown(
        options=OPTIONS, value=LABEL_BY_KEY.get(_saved.get("candidate")),
        label="Which is the derivative?", searchable=True,
    )
    # Table cells: 5 independently-named widgets, not a list/dict -- same
    # reason the slider/candidate above are separately named, a shared
    # container wouldn't give per-widget reactivity. Bare (label="") --
    # the build cell draws each point's x above its own cell instead, so
    # this reads as a real two-row table (x row, dy/dx row) rather than 5
    # individually-captioned boxes. Every point is graded independently
    # (not just one target column).
    sin_table_0 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[0]) if isinstance(_saved_table[0], (int, float)) else 0.0,
        label="",
    )
    sin_table_1 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[1]) if isinstance(_saved_table[1], (int, float)) else 0.0,
        label="",
    )
    sin_table_2 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[2]) if isinstance(_saved_table[2], (int, float)) else 0.0,
        label="",
    )
    sin_table_3 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[3]) if isinstance(_saved_table[3], (int, float)) else 0.0,
        label="",
    )
    sin_table_4 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[4]) if isinstance(_saved_table[4], (int, float)) else 0.0,
        label="",
    )
    return (
        sin_candidate,
        sin_slider,
        sin_table_0,
        sin_table_1,
        sin_table_2,
        sin_table_3,
        sin_table_4,
    )


@app.cell(hide_code=True)
def _(mo):
    get_sin_trace, set_sin_trace = mo.state({"xs": [], "ys": []})
    return get_sin_trace, set_sin_trace


@app.cell(hide_code=True)
def _(
    DX,
    FUNCTIONS,
    TABLE_POINTS_BY_KEY,
    build_function_plot,
    get_sin_trace,
    mo,
    set_sin_trace,
    sin_candidate,
    sin_slider,
    sin_table_0,
    sin_table_1,
    sin_table_2,
    sin_table_3,
    sin_table_4,
):
    _entry = next(e for e in FUNCTIONS if e[0] == "sin")
    sin_x = float(sin_slider.value)
    sin_candidate_value = sin_candidate.value

    _f = _entry[2]
    _true_deriv = (_f(sin_x + DX) - _f(sin_x)) / DX
    _trace = get_sin_trace()
    if sin_x not in _trace["xs"]:
        _trace = {"xs": _trace["xs"] + [sin_x], "ys": _trace["ys"] + [_true_deriv]}
        set_sin_trace(_trace)

    sin_table_values = [
        float(sin_table_0.value or 0.0), float(sin_table_1.value or 0.0), float(sin_table_2.value or 0.0),
        float(sin_table_3.value or 0.0), float(sin_table_4.value or 0.0),
    ]
    # Every table point is graded independently -- not just one target x
    # (the slider's position is never checked either). Each point's true
    # derivative is computed fresh from its OWN x, never from the
    # slider's current position (_true_deriv above is separate, only
    # used for the live trace plot).
    #
    # bool(...) wrapper: entries of sin_point_true_derivs are
    # numpy.float64 for every function using a numpy ufunc (sqrt, sin,
    # tan, exp, log, arcsin), which makes the comparison a numpy.bool_
    # rather than a plain bool -- and numpy.bool_ isn't JSON-serializable.
    #
    # Computed fresh every time from the table's CURRENT values -- no
    # persisted "once true, stays true" flags. A cell already keeps
    # whatever the student last typed regardless of slider movement, so
    # there's nothing to remember beyond that; a persisted flag would
    # (and did) let a since-overwritten wrong guess still read as correct
    # because it had been right at some earlier point.
    sin_point_true_derivs = [
        (_f(_p + DX) - _f(_p)) / DX for _p in TABLE_POINTS_BY_KEY["sin"]
    ]
    sin_point_achieved = [
        bool(abs(sin_table_values[_i] - sin_point_true_derivs[_i]) < 0.01)
        for _i in range(5)
    ]

    _plots = build_function_plot(
        _entry, FUNCTIONS, sin_x, sin_candidate_value, _trace["xs"], _trace["ys"],
        TABLE_POINTS_BY_KEY["sin"], sin_table_values,
    )
    # Shown so a report entry like "sin_deriv_value_0" can be traced
    # back to this tab.
    _key_label = mo.md("`sin`")
    _x_label = mo.md(f"**x = {sin_x:.4g}**")
    _table_points = TABLE_POINTS_BY_KEY["sin"]
    _table_widgets = [
        sin_table_0, sin_table_1, sin_table_2, sin_table_3, sin_table_4,
    ]
    _table_columns = [mo.vstack([mo.md("**x**"), mo.md("**dy/dx**")])]
    for _point, _widget in zip(_table_points, _table_widgets):
        _table_columns.append(mo.vstack([mo.md(f"**{_point:g}**"), _widget]))
    _table_row = mo.hstack(_table_columns, justify="start", gap=1)
    sin_tab = mo.vstack(
        [
            _key_label, _x_label, sin_slider, _plots,
            mo.md("Fill in `dy/dx` at every point in the table -- all 5 are graded:"),
            _table_row, sin_candidate,
        ],
        gap=1,
    )
    return (
        sin_candidate_value,
        sin_point_achieved,
        sin_point_true_derivs,
        sin_tab,
        sin_table_values,
        sin_x,
    )


@app.cell(hide_code=True)
def _(FUNCTIONS, LABEL_BY_KEY, OPTIONS, SAVED_STATE, mo, slider_start_value):
    _entry = next(e for e in FUNCTIONS if e[0] == "tan")
    _saved = SAVED_STATE.get("tan", {})
    _saved_table = _saved.get("table")
    if not isinstance(_saved_table, list) or len(_saved_table) != 5:
        _saved_table = [None] * 5
    tan_slider = mo.ui.slider(
        start=_entry[5][0], stop=_entry[5][1], step=0.05,
        value=slider_start_value(_entry, SAVED_STATE),
        label="x (slider position isn't graded -- fill in the table below)",
        full_width=True, show_value=True,
    )
    tan_candidate = mo.ui.dropdown(
        options=OPTIONS, value=LABEL_BY_KEY.get(_saved.get("candidate")),
        label="Which is the derivative?", searchable=True,
    )
    # Table cells: 5 independently-named widgets, not a list/dict -- same
    # reason the slider/candidate above are separately named, a shared
    # container wouldn't give per-widget reactivity. Bare (label="") --
    # the build cell draws each point's x above its own cell instead, so
    # this reads as a real two-row table (x row, dy/dx row) rather than 5
    # individually-captioned boxes. Every point is graded independently
    # (not just one target column).
    tan_table_0 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[0]) if isinstance(_saved_table[0], (int, float)) else 0.0,
        label="",
    )
    tan_table_1 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[1]) if isinstance(_saved_table[1], (int, float)) else 0.0,
        label="",
    )
    tan_table_2 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[2]) if isinstance(_saved_table[2], (int, float)) else 0.0,
        label="",
    )
    tan_table_3 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[3]) if isinstance(_saved_table[3], (int, float)) else 0.0,
        label="",
    )
    tan_table_4 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[4]) if isinstance(_saved_table[4], (int, float)) else 0.0,
        label="",
    )
    return (
        tan_candidate,
        tan_slider,
        tan_table_0,
        tan_table_1,
        tan_table_2,
        tan_table_3,
        tan_table_4,
    )


@app.cell(hide_code=True)
def _(mo):
    get_tan_trace, set_tan_trace = mo.state({"xs": [], "ys": []})
    return get_tan_trace, set_tan_trace


@app.cell(hide_code=True)
def _(
    DX,
    FUNCTIONS,
    TABLE_POINTS_BY_KEY,
    build_function_plot,
    get_tan_trace,
    mo,
    set_tan_trace,
    tan_candidate,
    tan_slider,
    tan_table_0,
    tan_table_1,
    tan_table_2,
    tan_table_3,
    tan_table_4,
):
    _entry = next(e for e in FUNCTIONS if e[0] == "tan")
    tan_x = float(tan_slider.value)
    tan_candidate_value = tan_candidate.value

    _f = _entry[2]
    _true_deriv = (_f(tan_x + DX) - _f(tan_x)) / DX
    _trace = get_tan_trace()
    if tan_x not in _trace["xs"]:
        _trace = {"xs": _trace["xs"] + [tan_x], "ys": _trace["ys"] + [_true_deriv]}
        set_tan_trace(_trace)

    tan_table_values = [
        float(tan_table_0.value or 0.0), float(tan_table_1.value or 0.0), float(tan_table_2.value or 0.0),
        float(tan_table_3.value or 0.0), float(tan_table_4.value or 0.0),
    ]
    # Every table point is graded independently -- not just one target x
    # (the slider's position is never checked either). Each point's true
    # derivative is computed fresh from its OWN x, never from the
    # slider's current position (_true_deriv above is separate, only
    # used for the live trace plot).
    #
    # bool(...) wrapper: entries of tan_point_true_derivs are
    # numpy.float64 for every function using a numpy ufunc (sqrt, sin,
    # tan, exp, log, arcsin), which makes the comparison a numpy.bool_
    # rather than a plain bool -- and numpy.bool_ isn't JSON-serializable.
    #
    # Computed fresh every time from the table's CURRENT values -- no
    # persisted "once true, stays true" flags. A cell already keeps
    # whatever the student last typed regardless of slider movement, so
    # there's nothing to remember beyond that; a persisted flag would
    # (and did) let a since-overwritten wrong guess still read as correct
    # because it had been right at some earlier point.
    tan_point_true_derivs = [
        (_f(_p + DX) - _f(_p)) / DX for _p in TABLE_POINTS_BY_KEY["tan"]
    ]
    tan_point_achieved = [
        bool(abs(tan_table_values[_i] - tan_point_true_derivs[_i]) < 0.01)
        for _i in range(5)
    ]

    _plots = build_function_plot(
        _entry, FUNCTIONS, tan_x, tan_candidate_value, _trace["xs"], _trace["ys"],
        TABLE_POINTS_BY_KEY["tan"], tan_table_values,
    )
    # Shown so a report entry like "tan_deriv_value_0" can be traced
    # back to this tab.
    _key_label = mo.md("`tan`")
    _x_label = mo.md(f"**x = {tan_x:.4g}**")
    _table_points = TABLE_POINTS_BY_KEY["tan"]
    _table_widgets = [
        tan_table_0, tan_table_1, tan_table_2, tan_table_3, tan_table_4,
    ]
    _table_columns = [mo.vstack([mo.md("**x**"), mo.md("**dy/dx**")])]
    for _point, _widget in zip(_table_points, _table_widgets):
        _table_columns.append(mo.vstack([mo.md(f"**{_point:g}**"), _widget]))
    _table_row = mo.hstack(_table_columns, justify="start", gap=1)
    tan_tab = mo.vstack(
        [
            _key_label, _x_label, tan_slider, _plots,
            mo.md("Fill in `dy/dx` at every point in the table -- all 5 are graded:"),
            _table_row, tan_candidate,
        ],
        gap=1,
    )
    return (
        tan_candidate_value,
        tan_point_achieved,
        tan_point_true_derivs,
        tan_tab,
        tan_table_values,
        tan_x,
    )


@app.cell(hide_code=True)
def _(FUNCTIONS, LABEL_BY_KEY, OPTIONS, SAVED_STATE, mo, slider_start_value):
    _entry = next(e for e in FUNCTIONS if e[0] == "exp")
    _saved = SAVED_STATE.get("exp", {})
    _saved_table = _saved.get("table")
    if not isinstance(_saved_table, list) or len(_saved_table) != 5:
        _saved_table = [None] * 5
    exp_slider = mo.ui.slider(
        start=_entry[5][0], stop=_entry[5][1], step=0.05,
        value=slider_start_value(_entry, SAVED_STATE),
        label="x (slider position isn't graded -- fill in the table below)",
        full_width=True, show_value=True,
    )
    exp_candidate = mo.ui.dropdown(
        options=OPTIONS, value=LABEL_BY_KEY.get(_saved.get("candidate")),
        label="Which is the derivative?", searchable=True,
    )
    # Table cells: 5 independently-named widgets, not a list/dict -- same
    # reason the slider/candidate above are separately named, a shared
    # container wouldn't give per-widget reactivity. Bare (label="") --
    # the build cell draws each point's x above its own cell instead, so
    # this reads as a real two-row table (x row, dy/dx row) rather than 5
    # individually-captioned boxes. Every point is graded independently
    # (not just one target column).
    exp_table_0 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[0]) if isinstance(_saved_table[0], (int, float)) else 0.0,
        label="",
    )
    exp_table_1 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[1]) if isinstance(_saved_table[1], (int, float)) else 0.0,
        label="",
    )
    exp_table_2 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[2]) if isinstance(_saved_table[2], (int, float)) else 0.0,
        label="",
    )
    exp_table_3 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[3]) if isinstance(_saved_table[3], (int, float)) else 0.0,
        label="",
    )
    exp_table_4 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[4]) if isinstance(_saved_table[4], (int, float)) else 0.0,
        label="",
    )
    return (
        exp_candidate,
        exp_slider,
        exp_table_0,
        exp_table_1,
        exp_table_2,
        exp_table_3,
        exp_table_4,
    )


@app.cell(hide_code=True)
def _(mo):
    get_exp_trace, set_exp_trace = mo.state({"xs": [], "ys": []})
    return get_exp_trace, set_exp_trace


@app.cell(hide_code=True)
def _(
    DX,
    FUNCTIONS,
    TABLE_POINTS_BY_KEY,
    build_function_plot,
    exp_candidate,
    exp_slider,
    exp_table_0,
    exp_table_1,
    exp_table_2,
    exp_table_3,
    exp_table_4,
    get_exp_trace,
    mo,
    set_exp_trace,
):
    _entry = next(e for e in FUNCTIONS if e[0] == "exp")
    exp_x = float(exp_slider.value)
    exp_candidate_value = exp_candidate.value

    _f = _entry[2]
    _true_deriv = (_f(exp_x + DX) - _f(exp_x)) / DX
    _trace = get_exp_trace()
    if exp_x not in _trace["xs"]:
        _trace = {"xs": _trace["xs"] + [exp_x], "ys": _trace["ys"] + [_true_deriv]}
        set_exp_trace(_trace)

    exp_table_values = [
        float(exp_table_0.value or 0.0), float(exp_table_1.value or 0.0), float(exp_table_2.value or 0.0),
        float(exp_table_3.value or 0.0), float(exp_table_4.value or 0.0),
    ]
    # Every table point is graded independently -- not just one target x
    # (the slider's position is never checked either). Each point's true
    # derivative is computed fresh from its OWN x, never from the
    # slider's current position (_true_deriv above is separate, only
    # used for the live trace plot).
    #
    # bool(...) wrapper: entries of exp_point_true_derivs are
    # numpy.float64 for every function using a numpy ufunc (sqrt, sin,
    # tan, exp, log, arcsin), which makes the comparison a numpy.bool_
    # rather than a plain bool -- and numpy.bool_ isn't JSON-serializable.
    #
    # Computed fresh every time from the table's CURRENT values -- no
    # persisted "once true, stays true" flags. A cell already keeps
    # whatever the student last typed regardless of slider movement, so
    # there's nothing to remember beyond that; a persisted flag would
    # (and did) let a since-overwritten wrong guess still read as correct
    # because it had been right at some earlier point.
    exp_point_true_derivs = [
        (_f(_p + DX) - _f(_p)) / DX for _p in TABLE_POINTS_BY_KEY["exp"]
    ]
    exp_point_achieved = [
        bool(abs(exp_table_values[_i] - exp_point_true_derivs[_i]) < 0.01)
        for _i in range(5)
    ]

    _plots = build_function_plot(
        _entry, FUNCTIONS, exp_x, exp_candidate_value, _trace["xs"], _trace["ys"],
        TABLE_POINTS_BY_KEY["exp"], exp_table_values,
    )
    # Shown so a report entry like "exp_deriv_value_0" can be traced
    # back to this tab.
    _key_label = mo.md("`exp`")
    _x_label = mo.md(f"**x = {exp_x:.4g}**")
    _table_points = TABLE_POINTS_BY_KEY["exp"]
    _table_widgets = [
        exp_table_0, exp_table_1, exp_table_2, exp_table_3, exp_table_4,
    ]
    _table_columns = [mo.vstack([mo.md("**x**"), mo.md("**dy/dx**")])]
    for _point, _widget in zip(_table_points, _table_widgets):
        _table_columns.append(mo.vstack([mo.md(f"**{_point:g}**"), _widget]))
    _table_row = mo.hstack(_table_columns, justify="start", gap=1)
    exp_tab = mo.vstack(
        [
            _key_label, _x_label, exp_slider, _plots,
            mo.md("Fill in `dy/dx` at every point in the table -- all 5 are graded:"),
            _table_row, exp_candidate,
        ],
        gap=1,
    )
    return (
        exp_candidate_value,
        exp_point_achieved,
        exp_point_true_derivs,
        exp_tab,
        exp_table_values,
        exp_x,
    )


@app.cell(hide_code=True)
def _(FUNCTIONS, LABEL_BY_KEY, OPTIONS, SAVED_STATE, mo, slider_start_value):
    _entry = next(e for e in FUNCTIONS if e[0] == "log")
    _saved = SAVED_STATE.get("log", {})
    _saved_table = _saved.get("table")
    if not isinstance(_saved_table, list) or len(_saved_table) != 5:
        _saved_table = [None] * 5
    log_slider = mo.ui.slider(
        start=_entry[5][0], stop=_entry[5][1], step=0.05,
        value=slider_start_value(_entry, SAVED_STATE),
        label="x (slider position isn't graded -- fill in the table below)",
        full_width=True, show_value=True,
    )
    log_candidate = mo.ui.dropdown(
        options=OPTIONS, value=LABEL_BY_KEY.get(_saved.get("candidate")),
        label="Which is the derivative?", searchable=True,
    )
    # Table cells: 5 independently-named widgets, not a list/dict -- same
    # reason the slider/candidate above are separately named, a shared
    # container wouldn't give per-widget reactivity. Bare (label="") --
    # the build cell draws each point's x above its own cell instead, so
    # this reads as a real two-row table (x row, dy/dx row) rather than 5
    # individually-captioned boxes. Every point is graded independently
    # (not just one target column).
    log_table_0 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[0]) if isinstance(_saved_table[0], (int, float)) else 0.0,
        label="",
    )
    log_table_1 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[1]) if isinstance(_saved_table[1], (int, float)) else 0.0,
        label="",
    )
    log_table_2 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[2]) if isinstance(_saved_table[2], (int, float)) else 0.0,
        label="",
    )
    log_table_3 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[3]) if isinstance(_saved_table[3], (int, float)) else 0.0,
        label="",
    )
    log_table_4 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[4]) if isinstance(_saved_table[4], (int, float)) else 0.0,
        label="",
    )
    return (
        log_candidate,
        log_slider,
        log_table_0,
        log_table_1,
        log_table_2,
        log_table_3,
        log_table_4,
    )


@app.cell(hide_code=True)
def _(mo):
    get_log_trace, set_log_trace = mo.state({"xs": [], "ys": []})
    return get_log_trace, set_log_trace


@app.cell(hide_code=True)
def _(
    DX,
    FUNCTIONS,
    TABLE_POINTS_BY_KEY,
    build_function_plot,
    get_log_trace,
    log_candidate,
    log_slider,
    log_table_0,
    log_table_1,
    log_table_2,
    log_table_3,
    log_table_4,
    mo,
    set_log_trace,
):
    _entry = next(e for e in FUNCTIONS if e[0] == "log")
    log_x = float(log_slider.value)
    log_candidate_value = log_candidate.value

    _f = _entry[2]
    _true_deriv = (_f(log_x + DX) - _f(log_x)) / DX
    _trace = get_log_trace()
    if log_x not in _trace["xs"]:
        _trace = {"xs": _trace["xs"] + [log_x], "ys": _trace["ys"] + [_true_deriv]}
        set_log_trace(_trace)

    log_table_values = [
        float(log_table_0.value or 0.0), float(log_table_1.value or 0.0), float(log_table_2.value or 0.0),
        float(log_table_3.value or 0.0), float(log_table_4.value or 0.0),
    ]
    # Every table point is graded independently -- not just one target x
    # (the slider's position is never checked either). Each point's true
    # derivative is computed fresh from its OWN x, never from the
    # slider's current position (_true_deriv above is separate, only
    # used for the live trace plot).
    #
    # bool(...) wrapper: entries of log_point_true_derivs are
    # numpy.float64 for every function using a numpy ufunc (sqrt, sin,
    # tan, exp, log, arcsin), which makes the comparison a numpy.bool_
    # rather than a plain bool -- and numpy.bool_ isn't JSON-serializable.
    #
    # Computed fresh every time from the table's CURRENT values -- no
    # persisted "once true, stays true" flags. A cell already keeps
    # whatever the student last typed regardless of slider movement, so
    # there's nothing to remember beyond that; a persisted flag would
    # (and did) let a since-overwritten wrong guess still read as correct
    # because it had been right at some earlier point.
    log_point_true_derivs = [
        (_f(_p + DX) - _f(_p)) / DX for _p in TABLE_POINTS_BY_KEY["log"]
    ]
    log_point_achieved = [
        bool(abs(log_table_values[_i] - log_point_true_derivs[_i]) < 0.01)
        for _i in range(5)
    ]

    _plots = build_function_plot(
        _entry, FUNCTIONS, log_x, log_candidate_value, _trace["xs"], _trace["ys"],
        TABLE_POINTS_BY_KEY["log"], log_table_values,
    )
    # Shown so a report entry like "log_deriv_value_0" can be traced
    # back to this tab.
    _key_label = mo.md("`log`")
    _x_label = mo.md(f"**x = {log_x:.4g}**")
    _table_points = TABLE_POINTS_BY_KEY["log"]
    _table_widgets = [
        log_table_0, log_table_1, log_table_2, log_table_3, log_table_4,
    ]
    _table_columns = [mo.vstack([mo.md("**x**"), mo.md("**dy/dx**")])]
    for _point, _widget in zip(_table_points, _table_widgets):
        _table_columns.append(mo.vstack([mo.md(f"**{_point:g}**"), _widget]))
    _table_row = mo.hstack(_table_columns, justify="start", gap=1)
    log_tab = mo.vstack(
        [
            _key_label, _x_label, log_slider, _plots,
            mo.md("Fill in `dy/dx` at every point in the table -- all 5 are graded:"),
            _table_row, log_candidate,
        ],
        gap=1,
    )
    return (
        log_candidate_value,
        log_point_achieved,
        log_point_true_derivs,
        log_tab,
        log_table_values,
        log_x,
    )


@app.cell(hide_code=True)
def _(FUNCTIONS, LABEL_BY_KEY, OPTIONS, SAVED_STATE, mo, slider_start_value):
    _entry = next(e for e in FUNCTIONS if e[0] == "arcsin")
    _saved = SAVED_STATE.get("arcsin", {})
    _saved_table = _saved.get("table")
    if not isinstance(_saved_table, list) or len(_saved_table) != 5:
        _saved_table = [None] * 5
    arcsin_slider = mo.ui.slider(
        start=_entry[5][0], stop=_entry[5][1], step=0.05,
        value=slider_start_value(_entry, SAVED_STATE),
        label="x (slider position isn't graded -- fill in the table below)",
        full_width=True, show_value=True,
    )
    arcsin_candidate = mo.ui.dropdown(
        options=OPTIONS, value=LABEL_BY_KEY.get(_saved.get("candidate")),
        label="Which is the derivative?", searchable=True,
    )
    # Table cells: 5 independently-named widgets, not a list/dict -- same
    # reason the slider/candidate above are separately named, a shared
    # container wouldn't give per-widget reactivity. Bare (label="") --
    # the build cell draws each point's x above its own cell instead, so
    # this reads as a real two-row table (x row, dy/dx row) rather than 5
    # individually-captioned boxes. Every point is graded independently
    # (not just one target column).
    arcsin_table_0 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[0]) if isinstance(_saved_table[0], (int, float)) else 0.0,
        label="",
    )
    arcsin_table_1 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[1]) if isinstance(_saved_table[1], (int, float)) else 0.0,
        label="",
    )
    arcsin_table_2 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[2]) if isinstance(_saved_table[2], (int, float)) else 0.0,
        label="",
    )
    arcsin_table_3 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[3]) if isinstance(_saved_table[3], (int, float)) else 0.0,
        label="",
    )
    arcsin_table_4 = mo.ui.number(
        start=-1000.0, stop=1000.0, step=0.0001,
        value=float(_saved_table[4]) if isinstance(_saved_table[4], (int, float)) else 0.0,
        label="",
    )
    return (
        arcsin_candidate,
        arcsin_slider,
        arcsin_table_0,
        arcsin_table_1,
        arcsin_table_2,
        arcsin_table_3,
        arcsin_table_4,
    )


@app.cell(hide_code=True)
def _(mo):
    get_arcsin_trace, set_arcsin_trace = mo.state({"xs": [], "ys": []})
    return get_arcsin_trace, set_arcsin_trace


@app.cell(hide_code=True)
def _(
    DX,
    FUNCTIONS,
    TABLE_POINTS_BY_KEY,
    arcsin_candidate,
    arcsin_slider,
    arcsin_table_0,
    arcsin_table_1,
    arcsin_table_2,
    arcsin_table_3,
    arcsin_table_4,
    build_function_plot,
    get_arcsin_trace,
    mo,
    set_arcsin_trace,
):
    _entry = next(e for e in FUNCTIONS if e[0] == "arcsin")
    arcsin_x = float(arcsin_slider.value)
    arcsin_candidate_value = arcsin_candidate.value

    _f = _entry[2]
    _true_deriv = (_f(arcsin_x + DX) - _f(arcsin_x)) / DX
    _trace = get_arcsin_trace()
    if arcsin_x not in _trace["xs"]:
        _trace = {"xs": _trace["xs"] + [arcsin_x], "ys": _trace["ys"] + [_true_deriv]}
        set_arcsin_trace(_trace)

    arcsin_table_values = [
        float(arcsin_table_0.value or 0.0), float(arcsin_table_1.value or 0.0), float(arcsin_table_2.value or 0.0),
        float(arcsin_table_3.value or 0.0), float(arcsin_table_4.value or 0.0),
    ]
    # Every table point is graded independently -- not just one target x
    # (the slider's position is never checked either). Each point's true
    # derivative is computed fresh from its OWN x, never from the
    # slider's current position (_true_deriv above is separate, only
    # used for the live trace plot).
    #
    # bool(...) wrapper: entries of arcsin_point_true_derivs are
    # numpy.float64 for every function using a numpy ufunc (sqrt, sin,
    # tan, exp, log, arcsin), which makes the comparison a numpy.bool_
    # rather than a plain bool -- and numpy.bool_ isn't JSON-serializable.
    #
    # Computed fresh every time from the table's CURRENT values -- no
    # persisted "once true, stays true" flags. A cell already keeps
    # whatever the student last typed regardless of slider movement, so
    # there's nothing to remember beyond that; a persisted flag would
    # (and did) let a since-overwritten wrong guess still read as correct
    # because it had been right at some earlier point.
    arcsin_point_true_derivs = [
        (_f(_p + DX) - _f(_p)) / DX for _p in TABLE_POINTS_BY_KEY["arcsin"]
    ]
    arcsin_point_achieved = [
        bool(abs(arcsin_table_values[_i] - arcsin_point_true_derivs[_i]) < 0.01)
        for _i in range(5)
    ]

    _plots = build_function_plot(
        _entry, FUNCTIONS, arcsin_x, arcsin_candidate_value, _trace["xs"], _trace["ys"],
        TABLE_POINTS_BY_KEY["arcsin"], arcsin_table_values,
    )
    # Shown so a report entry like "arcsin_deriv_value_0" can be traced
    # back to this tab.
    _key_label = mo.md("`arcsin`")
    _x_label = mo.md(f"**x = {arcsin_x:.4g}**")
    _table_points = TABLE_POINTS_BY_KEY["arcsin"]
    _table_widgets = [
        arcsin_table_0, arcsin_table_1, arcsin_table_2, arcsin_table_3, arcsin_table_4,
    ]
    _table_columns = [mo.vstack([mo.md("**x**"), mo.md("**dy/dx**")])]
    for _point, _widget in zip(_table_points, _table_widgets):
        _table_columns.append(mo.vstack([mo.md(f"**{_point:g}**"), _widget]))
    _table_row = mo.hstack(_table_columns, justify="start", gap=1)
    arcsin_tab = mo.vstack(
        [
            _key_label, _x_label, arcsin_slider, _plots,
            mo.md("Fill in `dy/dx` at every point in the table -- all 5 are graded:"),
            _table_row, arcsin_candidate,
        ],
        gap=1,
    )
    return (
        arcsin_candidate_value,
        arcsin_point_achieved,
        arcsin_point_true_derivs,
        arcsin_tab,
        arcsin_table_values,
        arcsin_x,
    )


@app.cell(hide_code=True)
def _(
    FUNCTION_LABEL_BY_KEY,
    arcsin_tab,
    cube_tab,
    exp_tab,
    log_tab,
    mo,
    sin_tab,
    sqrt_tab,
    square_tab,
    tan_tab,
):
    # Cheap to rerun on every change (just re-collects already-built tab
    # content into a dict) -- the expensive matplotlib work only happens
    # in whichever single function's build cell actually owns the widget
    # that changed.
    #
    # Titled by the function being explored (e.g. "x²"), not its
    # derivative -- FUNCTION_LABEL_BY_KEY, not LABEL_BY_KEY (which is the
    # derivative's label, used elsewhere for the dropdown options).
    function_tabs = mo.ui.tabs({
        FUNCTION_LABEL_BY_KEY["square"]: square_tab,
        FUNCTION_LABEL_BY_KEY["cube"]: cube_tab,
        FUNCTION_LABEL_BY_KEY["sqrt"]: sqrt_tab,
        FUNCTION_LABEL_BY_KEY["sin"]: sin_tab,
        FUNCTION_LABEL_BY_KEY["tan"]: tan_tab,
        FUNCTION_LABEL_BY_KEY["exp"]: exp_tab,
        FUNCTION_LABEL_BY_KEY["log"]: log_tab,
        FUNCTION_LABEL_BY_KEY["arcsin"]: arcsin_tab,
    })
    function_tabs
    return


@app.cell(hide_code=True)
def _(
    FUNCTIONS,
    LABEL_BY_KEY,
    arcsin_candidate_value,
    arcsin_point_achieved,
    arcsin_point_true_derivs,
    arcsin_table_values,
    arcsin_x,
    cube_candidate_value,
    cube_point_achieved,
    cube_point_true_derivs,
    cube_table_values,
    cube_x,
    exp_candidate_value,
    exp_point_achieved,
    exp_point_true_derivs,
    exp_table_values,
    exp_x,
    json,
    log_candidate_value,
    log_point_achieved,
    log_point_true_derivs,
    log_table_values,
    log_x,
    np,
    sin_candidate_value,
    sin_point_achieved,
    sin_point_true_derivs,
    sin_table_values,
    sin_x,
    sqrt_candidate_value,
    sqrt_point_achieved,
    sqrt_point_true_derivs,
    sqrt_table_values,
    sqrt_x,
    square_candidate_value,
    square_point_achieved,
    square_point_true_derivs,
    square_table_values,
    square_x,
    tan_candidate_value,
    tan_point_achieved,
    tan_point_true_derivs,
    tan_table_values,
    tan_x,
    utils,
):
    X_BY_KEY = {
        "square": square_x, "cube": cube_x, "sqrt": sqrt_x, "sin": sin_x,
        "tan": tan_x, "exp": exp_x, "log": log_x, "arcsin": arcsin_x,
    }
    CANDIDATE_BY_KEY = {
        "square": square_candidate_value, "cube": cube_candidate_value, "sqrt": sqrt_candidate_value,
        "sin": sin_candidate_value, "tan": tan_candidate_value, "exp": exp_candidate_value,
        "log": log_candidate_value, "arcsin": arcsin_candidate_value,
    }
    # Whether dy/dx was ever entered correctly at each of the 5 table
    # points independently (not wherever the slider happened to be at the
    # time) -- computed fresh from the table's current values every time
    # (see the per-function build cells). The slider's position itself is
    # never checked, only the values entered against each point's own
    # derivative. One list of 5 booleans per function.
    POINT_ACHIEVED_BY_KEY = {
        "square": square_point_achieved, "cube": cube_point_achieved, "sqrt": sqrt_point_achieved,
        "sin": sin_point_achieved, "tan": tan_point_achieved, "exp": exp_point_achieved,
        "log": log_point_achieved, "arcsin": arcsin_point_achieved,
    }
    POINT_TRUE_DERIV_BY_KEY = {
        "square": square_point_true_derivs, "cube": cube_point_true_derivs, "sqrt": sqrt_point_true_derivs,
        "sin": sin_point_true_derivs, "tan": tan_point_true_derivs, "exp": exp_point_true_derivs,
        "log": log_point_true_derivs, "arcsin": arcsin_point_true_derivs,
    }
    # The by-hand table (task.md step 2) -- persisted so it survives a
    # reload, and doubles as the graded guess for every point (see
    # POINT_ACHIEVED_BY_KEY above).
    TABLE_BY_KEY = {
        "square": square_table_values, "cube": cube_table_values,
        "sqrt": sqrt_table_values, "sin": sin_table_values,
        "tan": tan_table_values, "exp": exp_table_values,
        "log": log_table_values, "arcsin": arcsin_table_values,
    }

    _state_to_save = {
        key: {
            "x": X_BY_KEY[key], "candidate": CANDIDATE_BY_KEY[key], "table": TABLE_BY_KEY[key],
        }
        for key, *_rest in FUNCTIONS
    }
    utils.checker.save_npz({"state": json.dumps(_state_to_save)}, "state.npz")

    # Graded on submission only: for each function, was dy/dx ever entered
    # correctly at each of the 5 table points (regardless of where the
    # slider is now), and is the picked candidate the true derivative.
    # 6 keys per function instead of one opaque solved_{key} boolean, so a
    # wrong answer's report shows the actual numbers/labels involved:
    #   - "{key}_deriv_value_{i}": the derivative guess at table point i,
    #     vs. that point's true derivative.
    #   - "{key}_candidate_value": the picked formula, vs. the correct one.
    # Each key's own pass/fail still matches the original 0.01-tolerance
    # achieved/candidate-correctness conditions exactly -- when correct,
    # the reported value is set equal to the correct one (trivial match);
    # only a wrong answer reveals a genuine mismatch.
    _answer = {}
    for _key, *_rest in FUNCTIONS:
        _table_values = TABLE_BY_KEY[_key]
        _true_derivs = POINT_TRUE_DERIV_BY_KEY[_key]
        _point_achieved = POINT_ACHIEVED_BY_KEY[_key]
        for _i in range(5):
            _deriv_value = _true_derivs[_i] if _point_achieved[_i] else _table_values[_i]
            _answer[f"{_key}_deriv_value_{_i}"] = np.array(_deriv_value)

        _candidate_value = CANDIDATE_BY_KEY[_key]
        _candidate_correct = _candidate_value == _key
        _correct_label = LABEL_BY_KEY[_key]
        _candidate_label = LABEL_BY_KEY[_candidate_value] if _candidate_value is not None else "no answer"
        _answer[f"{_key}_candidate_value"] = np.array(
            _correct_label if _candidate_correct else _candidate_label, dtype="U32"
        )

    utils.checker.save_npz(_answer, utils.checker.get_data_dir() / "answer.npz")

    return


if __name__ == "__main__":
    app.run()
