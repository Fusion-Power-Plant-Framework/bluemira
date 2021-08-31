# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
2-bit machine learning..
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.linalg import lstsq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from BLUEPRINT.utilities.plottools import mathify
from bluemira.base.look_and_feel import bluemira_print
from BLUEPRINT.base.typebase import typechecked


class Law:
    """
    Regression law base object.

    Parameters
    ----------
    dataframe: DataFrame
        The DataFrame object for which to build a regression Law
    targets: List[str]
        The list of column names in the DataFrame which are result columns.
        These are excluded from the fitting.
    target: str
        The result column name in the DataFrame to fit to
    constant: Union[float, None]
        The constant (if any) to add to the Law.
    """

    def __init__(self, dataframe, targets, target, constant=None):

        self.db_original = dataframe.copy()
        self.db = None
        self.file = None
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None

        self._s = None
        self._i = None
        self.c_test = None
        self.c_train = None
        self.constant = None
        self.target = None
        self.inputs = None
        self.variables = None
        self.model = None
        self.r_2 = None

        self.process_df()
        self.process_constant(constant, target)
        self.drop_uniques()
        self.set_targets(targets, target)
        self.train_test(split=None)  # Default is 0.8

    def __read_file(self, target):
        path = "C:/Code/embryo/embryo/Data"
        if ".xls" in self.file:
            self.db_original = pd.read_excel(path + "/" + self.file)
        elif ".csv" in self.file:
            self.db_original = pd.read_csv(
                path + "/" + self.file, skiprows=0, error_bad_lines=False
            )
        if target == "t_d":
            self.db_original = self.db_original[np.isfinite(self.db_original[target])]
        # self.db_original.drop(['Blank'], axis=1, inplace=True)
        # for i in ['t_dir', 't_indir', 't_detrit']:
        # Converts to hours to get more reasonable scienfitic numbers
        # self.db_original[i] = self.db_original[i].apply(lambda x: x/3600)

    def process_df(self):
        """
        Process a DataFrame, dropping any infinite or NaN value entries
        """
        self.db_original.replace(np.inf, np.nan, inplace=True)
        self.db_original.dropna(inplace=True)
        self.db = self.db_original

    def process_constant(self, constant, target):
        """
        Handle a leading constant to form an equation of the type: y = CONSTANT + f(X)
        """
        if constant is None:
            self.constant = np.zeros(len(self.db))
            self._s = ""
            return
        if constant not in self.db.columns:
            raise ValueError("Need a variable in the dataframe.")
        self._s = constant
        # Get index of column // flag
        self._i = list(self.db.columns).index(constant)
        self.constant = self.db[constant]
        # Drop column in X
        self.db.drop(constant, axis=1, inplace=True)
        # Subtract constants from targets (already know their contribution)
        self.db[target] = self.db[target] - self.constant

    def drop_uniques(self):
        """
        Drop variables in input matrix which are constant
        """
        df = self.db
        nun = df.apply(pd.Series.nunique)
        cols = nun[nun == 1].index
        self.db = df.drop(cols, axis=1)

    def set_targets(self, targets, target):
        """
        Remove the result columns from the DataFrame, and set the fitting target.

        Parameters
        ----------
        targets: List[str]
            The list of column names in the DataFrame which are result columns.
            These are excluded from the fitting.
        target: str
            The result column name in the DataFrame to fit to
        """
        inputs = self.db.copy()
        for targ in targets:
            inputs = inputs.drop(targ, axis=1)
        self.inputs = inputs
        self.target = self.db[target]
        self.variables = inputs.columns

    def train_test(self, split=None):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        split: Union[float, None]
            The split fraction between train and test. Defaults to 0.8
        """
        (x_train, x_test, y_train, y_test, idx_train, idx_test) = train_test_split(
            self.inputs, self.target, np.arange(len(self.inputs)), test_size=split
        )
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.c_test = self.constant[idx_test]
        self.c_train = self.constant[idx_train]

    def fit(self, x, y):
        """
        Fit the regression model to the data.
        """
        self.model.fit(x, y)

    def score(self, x, y, show=False):
        """
        Score the model.
        """
        score = self.model.score(x, y)
        if show:
            bluemira_print("R\u00b2 = {0:.2f}".format(score))
        return round(score, 2)

    def score_model(self, bounds=False):
        """
        Plot the model result with score.
        """
        y_pred, x = self._process_xy()
        f, ax = plt.subplots(1, 1)
        ax.scatter(x, y_pred)
        r = np.arange(0, max(x))
        ax.plot(r, r, "k", lw=2, ls="--")
        self.cheap_kdeplot(x, y_pred, ax)
        ax.annotate("$R^{{2}}$ = {0:.2f}".format(self.r_2), xy=(0, max(y_pred) * 0.8))
        ax.set_xlim(left=0)
        ax.set_ylim([0, max(y_pred)])
        ax.set_aspect("equal")

        lab = mathify(self.target.name)
        ax.set_xlabel(lab + r" True")
        ax.set_ylabel(lab + r" Predicted")
        if bounds:
            h = r + 0.5
            length = r - 0.5
            ax.plot(r, h, "k", linestyle="--")
            ax.plot(r, length, "k", linestyle="--")

    def _process_xy(self):
        raise NotImplementedError

    def predict(self, x, xc=None, logged_input=False, show=False):
        """
        List and np.array supported only internal
        Xc is the constant, if used
        External interfaces must use dicts
        """
        if type(x) is list:
            c = np.array(xc) if xc is not None else 0
            x = np.array(x)
        elif type(x) is dict:
            y = []
            for i, var in enumerate(self.x_train.columns):
                y.append(x[var])
            c = x[self._s] if hasattr(self, "_i") else np.zeros(len(y[0]))
            x = y
        else:
            c = np.array(xc) if xc is not None else 0
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if logged_input is False:
            # Only use if calling externally when using PowerLaw
            x = np.log(x)
            pred = np.exp(self.model.predict(x))
        elif logged_input is True:
            pred = self.model.predict(x)
        if show:
            bluemira_print("{0} = {1:.2f}".format(self.target.name, pred[0]))
        return pred, c

    @staticmethod
    def cheap_kdeplot(x, y, ax, nbins=50):
        """
        Kernel density estimate plot.
        """
        x, y = sorted(x)[1:-1], sorted(y)[1:-1]
        hist, xedges, yedges = np.histogram2d(x, y, bins=(nbins, nbins), normed=True)
        x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1, nbins))
        y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins, 1))
        pdf = hist * (x_bin_sizes * y_bin_sizes)
        x, y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
        z = pdf.T
        ax.contour(x, y, z, origin="lower", cmap=plt.cm.viridis)

    def corrcoeff(self):
        """
        Plot correlation coefficient.
        """
        f, ax = plt.subplots(1, 1)
        c = ax.matshow(self.db.corr(), cmap="viridis")
        cb = f.colorbar(c)
        cb.set_clim(-1, 1)
        labels = [""] + list(self.db.columns)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        return ax


class LinearLaw(Law):
    """
    Fits a multiple linear regression to a dataset

    Parameters
    ----------
    dataframe: DataFrame
        The DataFrame object for which to build a regression Law
    targets: List[str]
        The list of column names in the DataFrame which are result columns.
        These are excluded from the fitting.
    target: str
        The result column name in the DataFrame to fit to
    constant: Union[float, None]
        The constant (if any) to add to the Law.
    """

    def __init__(self, dataframe, targets, target, **kwargs):
        super().__init__(dataframe, targets, target, **kwargs)
        self.initial_fit(self.x_train, self.y_train)
        self.optimise_fit()

    def initial_fit(self, inputs, target):
        """
        Get an inital regression law fit.
        """
        self.model = LinearRegression(normalize=True)
        self.fit(inputs, target)
        self.r_2 = self.score(inputs, target)
        return self.r_2

    def optimise_fit(self):
        """
        Optimise the variables used in the fitting, dropping any that don't affect
        the fitting score.
        """
        self._optimise_fit(self.x_train, self.y_train)

    def _optimise_fit(self, inputs, target):
        self.model = LinearRegression(normalize=True)
        for var in self.variables:
            tinput = inputs.drop(var, axis=1)
            self.fit(tinput, target)
            tscore = self.score(tinput, target)
            if tscore >= self.r_2:
                inputs = inputs.drop(var, axis=1)
                self.x_test = self.x_test.drop(var, axis=1)
        self.fit(inputs, target)
        self.r_2 = self.score(inputs, target)
        self.x_train = inputs

    def _process_xy(self):
        y_pred, c = self.predict(self.x_test, xc=self.c_test, logged_input=True)
        return y_pred + c, np.array(self.y_test) + c

    def print_fit(self):
        """
        Prints the equation of the LinearLaw and the fitting R^2 score.
        """
        varss = self.x_train.columns
        powers = self.model.coef_
        k = " + " if self._s != "" else ""
        eq = "\n" + self.target.name + " = " + self._s + k  # constant variable
        eq += "{0:.9f}*".format(np.exp(self.model.intercept_))
        for i in range(len(varss)):
            eq += "{0:.2f}*{1}".format(powers[i], varss[i])
            if i != len(varss) - 1:
                eq += "+"
        eq += "\nR\u00b2 = {0:.2f}".format(self.r_2)
        bluemira_print(eq)


class PowerLaw(LinearLaw):
    """
    Fits a multi-linear regression power law to a dataset.
    By default optimises the power law by dropping variables one by one
    and determining if the R**2 metric (%.2f) decreases at all.
    Trains the power law on 80% of the data (by default), and tests the fit on
    the remaining 20%. The data is split randomly, meaning that every time the
    object is run, the power law (and fit quality) can change.

    Parameters
    ----------
    dataframe: DataFrame
        The DataFrame object for which to build a regression Law
    targets: List[str]
        The list of column names in the DataFrame which are result columns.
        These are excluded from the fitting.
    target: str
        The result column name in the DataFrame to fit to
    constant: Union[float, None]
        The constant (if any) to add to the Law.
    """

    def __init__(self, dataframe, targets, target, **kwargs):
        dataframe = np.log(dataframe)
        super().__init__(dataframe, targets, target, **kwargs)

    def _process_xy(self):
        # Need to exp it again because of logged_input
        y_pred, c = self.predict(self.x_test, logged_input=True)
        return np.exp(y_pred) + c, np.exp(self.y_test)

    def print_fit(self):
        """
        Prints the equation of the PowerLaw and the fitting R^2 score.
        """
        varss = self.x_train.columns
        powers = self.model.coef_
        k = " + " if self._s != "" else ""
        eq = "\n" + self.target.name + " = " + self._s + k  # constant variable
        eq += "{0:.9f}*".format(np.exp(self.model.intercept_))
        for i in range(len(varss)):
            eq += "{0}^{1:.2f}".format(varss[i], powers[i])
            if i != len(varss) - 1:
                eq += "*"
        eq += "\nR\u00b2 = {0:.2f}".format(self.r_2)
        bluemira_print(eq)


@typechecked
def surface_fit(x, y, z, order: int = 2, n_grid: int = 30):
    """
    Fit a polynomial surface to a 3-D data set.

    Parameters
    ----------
    x: np.array(n)
        The x values of the data set
    y: np.array(n)
        The y values of the data set
    z: np.array(n)
        The z values of the data set
    order: int
        The order of the fitting polynomial
    n_grid: int
        The number of gridding points to use on the x and y data

    Returns
    -------
    x2d: np.array(n_grid, n_grid)
        The gridded x data
    y2d: np.array(n_grid, n_grid)
        The gridded y data
    zz: np.array(n_grid, n_grid)
        The gridded z fit data
    coeffs: list
        The list of polynomial coefficents
    r2: float
        The R^2 score of the fit

    Notes
    -----
    The coefficients are ordered by power, and by x and y. For an order = 2
    polynomial, the resultant equation would be:

    \t:math:`c_{1}x^{2}+c_{2}y^{2}+c_{3}xy+c_{4}x+c_{5}y+c_{6}`
    """
    x, y, z = np.array(x), np.array(y), np.array(z)
    if len(x) != len(y) != len(z):
        raise ValueError("x, y, and z must be of equal length.")

    n = len(x)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Grid the x and y arrays
    x2d, y2d = np.meshgrid(
        np.linspace(x_min, x_max, n_grid), np.linspace(y_min, y_max, n_grid)
    )
    xx = x2d.flatten()
    yy = y2d.flatten()

    if order == 1:  # Linear
        eq_matrix = np.c_[x, y, np.ones(n)]
        coeffs, _, _, _ = lstsq(eq_matrix, z)

        zz = np.dot(np.c_[xx, yy, np.ones(xx.shape)], coeffs).reshape(x2d.shape)

    elif order == 2:  # Quadratic
        eq_matrix = np.c_[x ** 2, y ** 2, x * y, x, y, np.ones(n)]
        coeffs, _, _, _ = lstsq(eq_matrix, z)

        zz = np.dot(
            np.c_[xx ** 2, yy ** 2, xx * yy, xx, yy, np.ones(xx.shape)], coeffs
        ).reshape(x2d.shape)

    elif order == 3:  # Cubic
        eq_matrix = np.c_[
            x ** 3,
            y ** 3,
            x ** 2 * y,
            x * y ** 2,
            x ** 2,
            y ** 2,
            x * y,
            x,
            y,
            np.ones(n),
        ]
        coeffs, _, _, _ = lstsq(eq_matrix, z)

        zz = np.dot(
            np.c_[
                xx ** 3,
                yy ** 3,
                xx ** 2 * yy,
                xx * yy ** 2,
                xx ** 2,
                yy ** 2,
                xx * yy,
                xx,
                yy,
                np.ones(xx.shape),
            ],
            coeffs,
        ).reshape(x2d.shape)

    else:
        # Too lazy for polynomial expansion...
        raise ValueError("order must be 1, 2 or 3.")

    z_predicted = np.dot(eq_matrix, coeffs).reshape(n)
    return x2d, y2d, zz, coeffs, r2_score(z, z_predicted)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
