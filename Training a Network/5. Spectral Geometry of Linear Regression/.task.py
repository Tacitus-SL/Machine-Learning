import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import importlib
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

    import json

    import numpy as np

    GRADIENT_STEPS = 20
    MIN_EIGENVALUE = 0.1
    MAX_EIGENVALUE = 10.0
    BASE_EXTENT = 3.0
    CONTOUR_LEVELS = 12
    CONTOUR_POINTS = 181
    CONTOUR_COLORS = [
        "#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89",
        "#35b779", "#6ece58", "#b5de2b", "#fde725", "#f9c74f", "#f8961e",
    ]

    # Ported from the ipynb version's public/private seed pair (0/42) --
    # only 42 survives here since the public stage is dropped (see the
    # save cell at the bottom): it never gated the grading criterion
    # anyway, only which noise pattern the demo scatter shows.
    _seed = 42
    rng = np.random.default_rng(_seed)
    noise_samples = rng.normal(scale=0.12, size=(GRADIENT_STEPS, 2))
    contour_angles = np.linspace(0.0, 2.0 * np.pi, CONTOUR_POINTS)

    _state_path = "state.npz"
    if utils.checker.is_valid_npz(_state_path):
        try:
            SAVED_STATE = json.loads(str(utils.checker.load_answer(_state_path)["state"]))
        except Exception:
            SAVED_STATE = {}
    else:
        SAVED_STATE = {}
    return (
        BASE_EXTENT, CONTOUR_COLORS, CONTOUR_LEVELS, GRADIENT_STEPS,
        MAX_EIGENVALUE, MIN_EIGENVALUE, SAVED_STATE, contour_angles, json,
        mo, noise_samples, np, utils,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Spectral Geometry of Linear Regression

        Explore how the eigenvalues of $X^T X$ shape the loss landscape of linear regression, then choose eigenvalues that make gradient descent converge as fast as possible.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Spectral Geometry of Linear Regression

        Before analyzing linear regression carefully, it helps to see the
        geometry hidden inside $X^T X$.

        ## The canonical decomposition of $X^T X$

        For any real design matrix $X$, the matrix

        $$A = X^T X$$

        is symmetric and positive semidefinite. By the spectral theorem, it
        has an orthogonal eigendecomposition

        $$X^T X = Q \Lambda Q^T,$$

        often remembered informally as a rotation-reflection matrix $Q$, a
        diagonal scaling matrix $\Lambda$, and the inverse rotation $Q^T$.
        In this demo we write the diagonal matrix as $L$.

        The pieces have precise roles:

        - $Q$ is orthogonal: $Q^TQ = QQ^T = I$. Its columns are
          perpendicular unit eigenvectors.
        - $L = \operatorname{diag}(\lambda_1, \lambda_2)$ contains the
          eigenvalues of $X^T X$.
        - Because $X^T X$ is positive semidefinite, every eigenvalue is
          nonnegative.
        - The quadratic landscape $w^T X^T Xw$ is flat in directions with
          small eigenvalues and steep in directions with large eigenvalues.
        - If an eigenvalue is zero or very close to zero, the inverse or
          pseudo-inverse can strongly amplify even small noise.
        """
    )
    return


@app.cell(hide_code=True)
def _(
    BASE_EXTENT,
    CONTOUR_LEVELS,
    GRADIENT_STEPS,
    MIN_EIGENVALUE,
    contour_angles,
    noise_samples,
    np,
):
    # Direct port of the ipynb version's pure-numpy helpers -- unchanged
    # math, only the widget/rendering plumbing around them changed.
    def make_q(angle_degrees):
        theta = np.deg2rad(angle_degrees)
        first = np.array([np.cos(theta), np.sin(theta)])
        second = np.array([-np.sin(theta), np.cos(theta)])
        return np.column_stack([first, second])

    def make_xtx(angle_degrees, lambda_1, lambda_2):
        Q = make_q(angle_degrees)
        L = np.diag([lambda_1, lambda_2])
        return Q @ L @ Q.T, Q, L

    def positive_eigenvalue(eigenvalue):
        return max(float(eigenvalue), MIN_EIGENVALUE)

    def condition_number_from_lambdas(lambda_1, lambda_2):
        lambda_1 = positive_eigenvalue(lambda_1)
        lambda_2 = positive_eigenvalue(lambda_2)
        return max(lambda_1, lambda_2) / min(lambda_1, lambda_2)

    def gradient_descent_path(A, steps=GRADIENT_STEPS):
        w = np.array([2.7, -2.2], dtype=float)
        largest_eigenvalue = max(float(np.linalg.eigvalsh(A).max()), 1e-8)
        # Stability requires learning_rate < 1 / largest_eigenvalue; the
        # 0.46 factor keeps this well clear of that bound (roughly half
        # of the previous 0.92, itself already a fraction of the limit)
        # so individual steps read as smaller, more gradual movement.
        learning_rate = 0.46 / (2.0 * largest_eigenvalue)
        path = [w.copy()]
        for _ in range(steps):
            w = w - learning_rate * (2.0 * A @ w)
            path.append(w.copy())
        return np.array(path)

    def pseudo_inverse_noise_points(Q, L):
        X = np.sqrt(L) @ Q.T
        return np.array([np.linalg.pinv(X) @ eps for eps in noise_samples])

    def contour_points(Q, lambda_1, lambda_2, visible_extent):
        max_level = 0.52 * visible_extent ** 2 * (lambda_1 + lambda_2)
        levels = np.linspace(max_level / CONTOUR_LEVELS, max_level, CONTOUR_LEVELS)
        cos_t = np.cos(contour_angles)
        sin_t = np.sin(contour_angles)
        curves = []
        for level in levels:
            eigen_coordinates = np.vstack([
                np.sqrt(level / lambda_1) * cos_t,
                np.sqrt(level / lambda_2) * sin_t,
            ])
            curves.append(Q @ eigen_coordinates)
        return curves

    def landscape_state(angle_degrees, lambda_1, lambda_2):
        lambda_1 = positive_eigenvalue(lambda_1)
        lambda_2 = positive_eigenvalue(lambda_2)
        reciprocals = np.array([1.0 / np.sqrt(lambda_1), 1.0 / np.sqrt(lambda_2)])
        A, Q, L = make_xtx(angle_degrees, lambda_1, lambda_2)
        path = gradient_descent_path(A)
        noise_points = pseudo_inverse_noise_points(Q, L)
        visible_extent = max(
            BASE_EXTENT,
            1.18 * float(reciprocals.max()),
            1.08 * float(np.abs(path).max()),
            1.08 * float(np.abs(noise_points).max()),
        )
        curves = contour_points(Q, lambda_1, lambda_2, visible_extent)
        condition = condition_number_from_lambdas(lambda_1, lambda_2)
        title = f"lambda1={lambda_1:.2f}, lambda2={lambda_2:.2f}, condition={condition:.2f}"
        return Q, np.diag(L), reciprocals, path, noise_points, visible_extent, curves, title

    return condition_number_from_lambdas, landscape_state


@app.cell(hide_code=True)
def _(CONTOUR_COLORS, landscape_state, mo):
    def build_landscape_plot(angle_degrees, lambda_1, lambda_2):
        import io

        import matplotlib.pyplot as plt

        _dpi = 100
        (
            Q, eigenvalues, reciprocals, path, noise_points, visible_extent,
            curves, title,
        ) = landscape_state(angle_degrees, lambda_1, lambda_2)

        _fig, _ax = plt.subplots(figsize=(7.4, 6.2))
        _axis_colors = ["#e63946", "#4361ee"]

        for _curve, _color in zip(curves, CONTOUR_COLORS):
            _ax.plot(_curve[0], _curve[1], color=_color, linewidth=1.5, alpha=0.78)

        for _index, _color in enumerate(_axis_colors):
            _vector = Q[:, _index] * reciprocals[_index]
            _ax.plot(
                [-_vector[0], _vector[0]], [-_vector[1], _vector[1]],
                linestyle="--", linewidth=2.0, color=_color,
                label=f"1/sqrt(lambda) axis {_index + 1}: {reciprocals[_index]:.2f}",
            )

        for _index, _color in enumerate(_axis_colors):
            _direction = Q[:, _index]
            _ax.plot(
                [0.0, _direction[0]], [0.0, _direction[1]],
                marker="o", linewidth=4.0, markersize=7, color=_color,
                label=f"unit q{_index + 1}, lambda={eigenvalues[_index]:.2f}",
            )
            _label_point = _direction * (reciprocals[_index] + 0.18)
            _ax.text(
                _label_point[0], _label_point[1],
                f"1/sqrt(lambda{_index + 1})={reciprocals[_index]:.2f}",
                fontsize=10, color=_color,
            )

        _ax.plot(
            path[:, 0], path[:, 1], color="#fb8500", marker="o",
            linewidth=2.5, markersize=5, label="gradient descent",
        )
        _ax.scatter(
            noise_points[:, 0], noise_points[:, 1], s=36, color="#f72585",
            edgecolor="white", linewidth=0.7, alpha=0.82,
            label="pseudo-inverse noise",
        )
        _ax.scatter([0], [0], s=46, color="white", edgecolor="black", linewidth=1.0, zorder=5)

        _ax.set_xlabel("w1")
        _ax.set_ylabel("w2")
        _ax.set_aspect("equal", adjustable="box")
        _ax.grid(True, alpha=0.22)
        _ax.legend(loc="upper right", fontsize=7)
        _ax.set_xlim(-visible_extent, visible_extent)
        _ax.set_ylim(-visible_extent, visible_extent)
        _ax.set_title(title, fontsize=10)
        _fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.09)

        # Fixed-size image (buffer + explicit width/height, no
        # bbox_inches="tight"), same technique used everywhere else this
        # session -- mo.hstack/vstack would otherwise stretch/shrink an
        # embedded figure to fill whatever space is available.
        _buf = io.BytesIO()
        _fig.savefig(_buf, format="png", dpi=_dpi)
        _buf.seek(0)
        _width_px = round(_fig.get_figwidth() * _dpi)
        _height_px = round(_fig.get_figheight() * _dpi)
        plt.close(_fig)
        return mo.image(src=_buf, width=_width_px, height=_height_px)

    return (build_landscape_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Build $Q$, build $L$, and watch the landscape

        The angle control rotates the first eigenvector of the
        two-dimensional $Q$ matrix. The eigenvalue controls set
        $\lambda_1$ and $\lambda_2$ directly, so smaller values make that
        direction flatter and more noise-sensitive. Together they create

        $$X^T X = QLQ^T.$$

        The contour plot shows levels of $w^T X^T Xw$. The solid colored
        arrows are unit eigenvectors, and the dashed colored axes have
        length $1 / \sqrt{\lambda_i}$. The orange path shows 20 gradient
        descent steps on the same quadratic. The pink points show
        pseudo-inverse solutions to

        $$Xw = \varepsilon, \qquad w = (X^T X)^+X^T\varepsilon,$$

        where $\varepsilon$ is random noise -- one different $\varepsilon$
        vector per gradient descent step. When an eigenvalue is small,
        those errors can become surprisingly large weights.

        Use the controls to look for the eigenvalues that make gradient
        descent converge as fast as possible. The relevant warning light
        is the condition number

        $$\kappa = \frac{\max(\lambda_1, \lambda_2)}{\min(\lambda_1, \lambda_2)}.$$
        """
    )
    return


@app.cell(hide_code=True)
def _(MAX_EIGENVALUE, MIN_EIGENVALUE, SAVED_STATE, mo):
    angle_slider = mo.ui.slider(
        start=0.0, stop=360.0, step=5.0,
        value=float(SAVED_STATE.get("angle_degrees", 35.0)),
        label="angle (degrees)", full_width=True, show_value=True,
    )
    lambda_1_slider = mo.ui.slider(
        start=MIN_EIGENVALUE, stop=MAX_EIGENVALUE, step=0.1,
        # Defaults deliberately unequal (not 1.0/1.0) so a fresh, untouched
        # widget doesn't already sit at the graded answer (equal
        # eigenvalues, condition_number == 1.0) -- see
        # compile-check-cleanup-check-cycle in
        # Courses/.agents/rules/verification-workflow.md.
        value=float(SAVED_STATE.get("lambda_1", 1.0)),
        label="lambda 1", full_width=True, show_value=True,
    )
    lambda_2_slider = mo.ui.slider(
        start=MIN_EIGENVALUE, stop=MAX_EIGENVALUE, step=0.1,
        value=float(SAVED_STATE.get("lambda_2", 3.0)),
        label="lambda 2", full_width=True, show_value=True,
    )
    return angle_slider, lambda_1_slider, lambda_2_slider


@app.cell(hide_code=True)
def _(
    angle_slider,
    build_landscape_plot,
    condition_number_from_lambdas,
    lambda_1_slider,
    lambda_2_slider,
    mo,
):
    angle_degrees = float(angle_slider.value)
    chosen_lambda_1 = float(lambda_1_slider.value)
    chosen_lambda_2 = float(lambda_2_slider.value)

    # No live right/wrong feedback -- the condition number is shown (it's
    # not the answer itself, just the quantity being minimized), but there
    # is no indication of whether the current value counts as "correct".
    condition_number = condition_number_from_lambdas(chosen_lambda_1, chosen_lambda_2)

    _plot = build_landscape_plot(angle_degrees, chosen_lambda_1, chosen_lambda_2)
    _status = mo.md(f"**condition number: {condition_number:.4g}**")

    landscape_tab = mo.vstack(
        [angle_slider, lambda_1_slider, lambda_2_slider, _plot, _status],
        gap=1,
    )
    return angle_degrees, chosen_lambda_1, chosen_lambda_2, condition_number, landscape_tab


@app.cell(hide_code=True)
def _(landscape_tab, mo):
    mo.md(
        r"""
        ## Task: make gradient descent as fast as possible

        For a two-dimensional quadratic loss with Hessian geometry
        controlled by eigenvalues $\lambda_1$ and $\lambda_2$, gradient
        descent converges fastest when the landscape is as round as
        possible. Use the sliders above to choose eigenvalues in
        $[0.1, 10]$ that make gradient descent as fast as possible --
        your answer is read directly from wherever the sliders are
        currently set (no code to write). There's no live right/wrong
        feedback; only the actual submission is graded on the condition
        number

        $$\kappa = \frac{\lambda_{max}}{\lambda_{min}}.$$
        """
    )
    return


@app.cell(hide_code=True)
def _(landscape_tab):
    landscape_tab
    return


@app.cell(hide_code=True)
def _(
    angle_degrees,
    chosen_lambda_1,
    chosen_lambda_2,
    condition_number,
    json,
    np,
    utils,
):
    _state_to_save = {
        "angle_degrees": angle_degrees,
        "lambda_1": chosen_lambda_1,
        "lambda_2": chosen_lambda_2,
    }
    utils.checker.save_npz({"state": json.dumps(_state_to_save)}, "state.npz")

    _answer = {"condition_number": np.array(float(condition_number))}
    utils.checker.save_npz(_answer, utils.checker.get_data_dir() / "answer.npz")

    # -- start --
    # _check_code_task() unconditionally requires inputs_private.npz to
    # exist right after the very first run of .task.py -- this task has
    # nothing that needs hiding (the grading criterion, equal eigenvalues,
    # is public knowledge), so a placeholder is all that's needed. No
    # inputs_public.npz: skipping the public stage means the report stays
    # pass/fail only, never revealing the correct condition number.
    utils.checker.save_npz(
        {"placeholder": np.array(0)},
        utils.checker.get_data_dir() / "inputs_private.npz",
    )
    # -- end --

    # -- start --
    # Force the reference condition number to 1.0 (equal eigenvalues --
    # the theoretically fastest-converging, best-conditioned landscape)
    # here instead of whatever the student's own saved state currently
    # is. This whole block is stripped out of task.py by the marker
    # convention, so the student's own run always keeps the live answer
    # above.
    utils.checker.save_npz(
        {"condition_number": np.array(1.0)},
        utils.checker.get_data_dir() / "answer.npz",
    )
    # -- end --
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Conclusion

        The eigenvalues of $X^T X$ control the practical difficulty of
        linear regression. Large differences between eigenvalues make the
        loss landscape long and narrow, so gradient descent may move
        slowly or awkwardly unless the step size is chosen with care.
        Very small eigenvalues also make the pseudo-inverse unstable: tiny
        noise in $Xw = \varepsilon$ can be amplified into large changes in
        $w$.

        So both optimization and closed-form solving depend on the same
        geometry. Gradient descent does not always work well, and even
        the pseudo-inverse is not always stable; the eigenvalues of
        $X^T X$ are the warning lights.
        """
    )
    return


if __name__ == "__main__":
    app.run()
