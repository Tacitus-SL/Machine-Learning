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

    # ORDERED_CARDS is the correct answer -- placeholder so this cell still
    # returns something defined if the marker block below is stripped
    # (student-facing task.py); the real value is only ever set inside it.

    # -- start --
    import numpy as np

    ORDERED_CARDS = np.array([
        r"""Consider a set of samples 

$$x_1, x_2, \dots, x_n$$

that are supposedly drawn from Normal Distribution:

$$p(x) = \frac{1}{\sqrt{2 \pi} \sigma} \exp(- \frac{(x - \mu)^2}{2\sigma^2} )$$

where $\sigma$ is standard deviation of $x$
and $\mu$ is mean

Thus the task is to find parameters $\sigma$ and $\mu$ that
the best fit for the set of samples""",
            r"""Let us find probability of the set of samples assuming that
every sample is independent on the others:

$$P(x_1, x_2, \dots, x_N) = \prod_k p(x_k)$$""",
            r"""Take logarithm of the probability of the observed set to get rid
of summation:

$$\log P(x_1, x_2, \dots, x_N) = $$
$$= \sum_k \log p(x_k) =$$
$$= - k \frac{1}{2} \log (2\pi) - k \log \sigma - \sum_k \frac{(x_k - \mu)^2}{2 \sigma^2}$$""",
            r"""The first term is a constant and cannot be optimized, while the
second and third ones can be maximized by tuning $\mu$ and $\Sigma$""",
            r"""Turning maximization problem into standard minimization problem,
we get the loss function:

$$L = k \log \sigma + \sum_k \frac{(x_k - \mu)^2}{2 \sigma^2}$$

that is surprisingly similar to the Squared Error (up to a scaling factor and an additional term)""",
            r"""The best parameters will be such $\Sigma$ and $\mu$ that are the 
solution of the following optization problem:

$$\sigma, \mu = \underset{\sigma, \mu}{\textrm{argmin}} \left[k \log \sigma + \sum_k \frac{(x_k - \mu)^2}{2 \sigma^2} \right]$$""",
            r"""Now one can find optimal value for $\mu$ by taking a derivative

$$\frac{\partial L}{\partial \mu} = \frac{1}{\sigma^2} \sum_k (x_k - \mu) = \frac{1}{2 \sigma^2} \sum_k x_k - \frac{k \mu}{2 \sigma^2}$$""",
            r"""Setting this derivative to zero (with accordance with extremum condition), we get equation:

$$\frac{1}{2 \sigma^2} \sum_k x_k - \frac{k \mu}{2 \sigma^2} = 0$$""",
            r"""That after simplification turns into

$$\mu = \frac{\sum_k x_k}{k}$$

I.e., optimal parameter $\mu$ for Normal Distribution is the mean of the observed samples""",
            r"""Consider now 

$$\frac{\partial L}{\partial \sigma} = 0$$""",
            r"""It is equal to 

$$\frac{k}{\sigma} - \sum_k \frac{(x_k - \mu)^2}{\sigma^3} = 0$$""",
            r"""After simplification, it becomes:

$$\sigma^2 = \frac{1}{k} \sum_k (x_k - \mu)^2$$

I.e., optimal parameter \sigma^2 for Normal Distribution is equal to the 
mean of squared deviations.""",
        ])
    utils.cards.generate_ordering_inputs(ORDERED_CARDS)
    # -- end --

    CARDS = utils.cards.load_ordering_cards()
    get_order, set_order = utils.cards.init_ordering_state(mo, CARDS)
    return CARDS, ORDERED_CARDS, get_order, mo, set_order, utils


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
def _(ORDERED_CARDS, down_buttons, up_buttons, utils, widget):
    # Referencing both arrays in another cell keeps their callbacks registered.
    _controls = down_buttons, up_buttons

    # -- start --
    utils.cards.save_ordering_answer(ORDERED_CARDS)
    # -- end --

    widget
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
