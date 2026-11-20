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

$$\mathbf x_1, \mathbf x_2, \dots, \mathbf x_N$$

that are supposedly drawn from Normal Distribution:

$$p(\mathbf x) = \frac{1}{(2 \pi)^{n/2} (\det \Sigma)^{1/2}} \exp\left(- \frac{1}{2} (\mathbf x - \mu)^T \Sigma^{-1}(\mathbf x - \mu) \right)$$

where $n$ is the dimension of vector $\mathbf x$, $\Sigma$ is the covariance
matrix of coordinates of $\mathbf x$ and $\mu$ is the vector of means

Thus the task is to find parameters $\Sigma$ and $\mu$ that
best fit the set of samples""",
            r"""Let us find probability of the set of samples assuming that
every sample is independent of the others:

$$P(\mathbf x_1, \mathbf x_2, \dots, \mathbf x_N) = \prod_k p(\mathbf x_k)$$""",
            r"""Take the logarithm of the probability of the observed set to turn
the product into a sum:

$$\log P(\mathbf x_1, \mathbf x_2, \dots, \mathbf x_N) = $$
$$= \sum_k \log p(\mathbf x_k) =$$
$$= - \frac{N n}{2} \log (2 \pi) - \frac{N}{2} \log (\det \Sigma) - \frac{1}{2} \sum_k (\mathbf x_k - \mu)^T \Sigma^{-1}(\mathbf x_k - \mu)$$""",
            r"""The first term is a constant and cannot be optimized, while the
second and third ones can be maximized by tuning $\mu$ and $\Sigma$""",
            r"""Turning maximization problem into standard minimization problem,
we get the loss function:

$$L = \frac{N}{2} \log (\det \Sigma) + \frac{1}{2} \sum_k (\mathbf x_k - \mu)^T \Sigma^{-1}(\mathbf x_k - \mu)$$""",
            r"""The best parameters will be such $\Sigma$ and $\mu$ that are the 
solution of the following optimization problem:

$$\Sigma, \mu = \underset{\Sigma, \mu}{\textrm{argmin}} \left[\frac{N}{2} \log (\det \Sigma) + \frac{1}{2} \sum_k (\mathbf x_k - \mu)^T \Sigma^{-1}(\mathbf x_k - \mu) \right]$$""",
            r"""Now one can find optimal value for $\mu$ by taking a derivative
(the matrix $\Sigma^{-1}$ is symmetric)

$$\frac{\partial L}{\partial \mu} = - \Sigma^{-1} \sum_k (\mathbf x_k - \mu) = - \Sigma^{-1} \sum_k \mathbf x_k + N \Sigma^{-1} \mu$$""",
            r"""Setting this derivative to zero (in accordance with the extremum condition), we get the equation:

$$- \Sigma^{-1} \sum_k \mathbf x_k + N \Sigma^{-1} \mu = 0$$""",
            r"""That after multiplying by $\Sigma$ from the left turns into

$$\mu = \frac{1}{N} \sum_k \mathbf x_k$$

I.e., optimal parameter $\mu$ for Normal Distribution is the mean of the observed samples""",
            r"""Consider now the derivative with respect to the inverse matrix
$\Sigma^{-1}$, using $\log (\det \Sigma) = - \log (\det \Sigma^{-1})$:

$$\frac{\partial L}{\partial \Sigma^{-1}} = 0$$""",
            r"""Using $\frac{\partial}{\partial A} \log (\det A) = A^{-T}$ and
$\frac{\partial}{\partial A} \mathbf v^T A \mathbf v = \mathbf v \mathbf v^T$, it is equal to

$$- \frac{N}{2} \Sigma + \frac{1}{2} \sum_k (\mathbf x_k - \mu)(\mathbf x_k - \mu)^T = 0$$""",
            r"""After simplification, it becomes:

$$\Sigma = \frac{1}{N} \sum_k (\mathbf x_k - \mu)(\mathbf x_k - \mu)^T$$

I.e., optimal parameter $\Sigma$ for Normal Distribution is equal to the
mean of outer products of deviations from the mean (the sample covariance matrix).""",
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
