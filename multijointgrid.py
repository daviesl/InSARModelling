from __future__ import division
from itertools import product
from distutils.version import LooseVersion
import warnings
from textwrap import dedent

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import utils
from .palettes import color_palette, blend_palette
from .external.six import string_types
from .distributions import distplot, kdeplot,  _freedman_diaconis_bins


__all__ = ["MulitJointGrid", "multijointplot"]

class MulitJointGrid(object):
    """Grid for drawing a multiple bivariate plots with marginal univariate plots."""

    def __init__(self, x, y, data=None, size=6, ratio=5, space=.2,
                 dropna=True, xlim=None, ylim=None):
        """Set up the grid of subplots.

        Parameters
        ----------
        x, y : strings or vectors
            Data or names of variables in ``data``.
        data : DataFrame, optional
            DataFrame when ``x`` and ``y`` are variable names.
        size : numeric
            Size of each side of the figure in inches (it will be square).
        ratio : numeric
            Ratio of joint axes size to marginal axes height.
        space : numeric, optional
            Space between the joint and marginal axes
        dropna : bool, optional
            If True, remove observations that are missing from `x` and `y`.
        {x, y}lim : two-tuples, optional
            Axis limits to set before plotting.

        See Also
        --------
        jointplot : High-level interface for drawing bivariate plots with
                    several different default plot kinds.

        Examples
        --------

        Initialize the figure but don't draw any plots onto it:

        .. plot::
            :context: close-figs

            >>> import seaborn as sns; sns.set(style="ticks", color_codes=True)
            >>> tips = sns.load_dataset("tips")
            >>> g = sns.JointGrid(x="total_bill", y="tip", data=tips)

        Add plots using default parameters:

        .. plot::
            :context: close-figs

            >>> g = sns.JointGrid(x="total_bill", y="tip", data=tips)
            >>> g = g.plot(sns.regplot, sns.distplot)

        Draw the join and marginal plots separately, which allows finer-level
        control other parameters:

        .. plot::
            :context: close-figs

            >>> import matplotlib.pyplot as plt
            >>> g = sns.JointGrid(x="total_bill", y="tip", data=tips)
            >>> g = g.plot_joint(plt.scatter, color=".5", edgecolor="white")
            >>> g = g.plot_marginals(sns.distplot, kde=False, color=".5")

        Draw the two marginal plots separately:

        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> g = sns.JointGrid(x="total_bill", y="tip", data=tips)
            >>> g = g.plot_joint(plt.scatter, color="m", edgecolor="white")
            >>> _ = g.ax_marg_x.hist(tips["total_bill"], color="b", alpha=.6,
            ...                      bins=np.arange(0, 60, 5))
            >>> _ = g.ax_marg_y.hist(tips["tip"], color="r", alpha=.6,
            ...                      orientation="horizontal",
            ...                      bins=np.arange(0, 12, 1))

        Add an annotation with a statistic summarizing the bivariate
        relationship:

        .. plot::
            :context: close-figs

            >>> from scipy import stats
            >>> g = sns.JointGrid(x="total_bill", y="tip", data=tips)
            >>> g = g.plot_joint(plt.scatter,
            ...                  color="g", s=40, edgecolor="white")
            >>> g = g.plot_marginals(sns.distplot, kde=False, color="g")
            >>> g = g.annotate(stats.pearsonr)

        Use a custom function and formatting for the annotation

        .. plot::
            :context: close-figs

            >>> g = sns.JointGrid(x="total_bill", y="tip", data=tips)
            >>> g = g.plot_joint(plt.scatter,
            ...                  color="g", s=40, edgecolor="white")
            >>> g = g.plot_marginals(sns.distplot, kde=False, color="g")
            >>> rsquare = lambda a, b: stats.pearsonr(a, b)[0] ** 2
            >>> g = g.annotate(rsquare, template="{stat}: {val:.2f}",
            ...                stat="$R^2$", loc="upper left", fontsize=12)

        Remove the space between the joint and marginal axes:

        .. plot::
            :context: close-figs

            >>> g = sns.JointGrid(x="total_bill", y="tip", data=tips, space=0)
            >>> g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
            >>> g = g.plot_marginals(sns.kdeplot, shade=True)

        Draw a smaller plot with relatively larger marginal axes:

        .. plot::
            :context: close-figs

            >>> g = sns.JointGrid(x="total_bill", y="tip", data=tips,
            ...                   size=5, ratio=2)
            >>> g = g.plot_joint(sns.kdeplot, cmap="Reds_d")
            >>> g = g.plot_marginals(sns.kdeplot, color="r", shade=True)

        Set limits on the axes:

        .. plot::
            :context: close-figs

            >>> g = sns.JointGrid(x="total_bill", y="tip", data=tips,
            ...                   xlim=(0, 50), ylim=(0, 8))
            >>> g = g.plot_joint(sns.kdeplot, cmap="Purples_d")
            >>> g = g.plot_marginals(sns.kdeplot, color="m", shade=True)

        """
        # Set up the subplot grid
        f = plt.figure(figsize=(size, size))
        gs = plt.GridSpec(ratio + 1, ratio + 1)

        ax_joint = f.add_subplot(gs[1:, :-1])
        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
        ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

        self.fig = f
        self.ax_joint = ax_joint
        self.ax_marg_x = ax_marg_x
        self.ax_marg_y = ax_marg_y

        # Turn off tick visibility for the measure axis on the marginal plots
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        # Turn off the ticks on the density axis for the marginal plots
        plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
        plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
        plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
        plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
        plt.setp(ax_marg_x.get_yticklabels(), visible=False)
        plt.setp(ax_marg_y.get_xticklabels(), visible=False)
        ax_marg_x.yaxis.grid(False)
        ax_marg_y.xaxis.grid(False)

        # Possibly extract the variables from a DataFrame
        if data is not None:
            x = data.get(x, x)
            y = data.get(y, y)

        for var in [x, y]:
            if isinstance(var, string_types):
                err = "Could not interpret input '{}'".format(var)
                raise ValueError(err)

        # Possibly drop NA
        if dropna:
            not_na = pd.notnull(x) & pd.notnull(y)
            x = x[not_na]
            y = y[not_na]

        # Find the names of the variables
        if hasattr(x, "name"):
            xlabel = x.name
            ax_joint.set_xlabel(xlabel)
        if hasattr(y, "name"):
            ylabel = y.name
            ax_joint.set_ylabel(ylabel)

        # Convert the x and y data to arrays for plotting
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        if xlim is not None:
            ax_joint.set_xlim(xlim)
        if ylim is not None:
            ax_joint.set_ylim(ylim)

        # Make the grid look nice
        utils.despine(f)
        utils.despine(ax=ax_marg_x, left=True)
        utils.despine(ax=ax_marg_y, bottom=True)
        f.tight_layout()
        f.subplots_adjust(hspace=space, wspace=space)

    def plot(self, joint_func, marginal_func, annot_func=None):
        """Shortcut to draw the full plot.

        Use `plot_joint` and `plot_marginals` directly for more control.

        Parameters
        ----------
        joint_func, marginal_func: callables
            Functions to draw the bivariate and univariate plots.

        Returns
        -------
        self : JointGrid instance
            Returns `self`.

        """
        self.plot_marginals(marginal_func)
        self.plot_joint(joint_func)
        if annot_func is not None:
            self.annotate(annot_func)
        return self

    def plot_joint(self, func, **kwargs):
        """Draw a bivariate plot of `x` and `y`.

        Parameters
        ----------
        func : plotting callable
            This must take two 1d arrays of data as the first two
            positional arguments, and it must plot on the "current" axes.
        kwargs : key, value mappings
            Keyword argument are passed to the plotting function.

        Returns
        -------
        self : JointGrid instance
            Returns `self`.

        """
        plt.sca(self.ax_joint)
        func(self.x, self.y, **kwargs)

        return self

    def plot_marginals(self, func, **kwargs):
        """Draw univariate plots for `x` and `y` separately.

        Parameters
        ----------
        func : plotting callable
            This must take a 1d array of data as the first positional
            argument, it must plot on the "current" axes, and it must
            accept a "vertical" keyword argument to orient the measure
            dimension of the plot vertically.
        kwargs : key, value mappings
            Keyword argument are passed to the plotting function.

        Returns
        -------
        self : JointGrid instance
            Returns `self`.

        """
        kwargs["vertical"] = False
        plt.sca(self.ax_marg_x)
        func(self.x, **kwargs)

        kwargs["vertical"] = True
        plt.sca(self.ax_marg_y)
        func(self.y, **kwargs)

        return self

    def annotate(self, func, template=None, stat=None, loc="best", **kwargs):
        """Annotate the plot with a statistic about the relationship.

        Parameters
        ----------
        func : callable
            Statistical function that maps the x, y vectors either to (val, p)
            or to val.
        template : string format template, optional
            The template must have the format keys "stat" and "val";
            if `func` returns a p value, it should also have the key "p".
        stat : string, optional
            Name to use for the statistic in the annotation, by default it
            uses the name of `func`.
        loc : string or int, optional
            Matplotlib legend location code; used to place the annotation.
        kwargs : key, value mappings
            Other keyword arguments are passed to `ax.legend`, which formats
            the annotation.

        Returns
        -------
        self : JointGrid instance.
            Returns `self`.

        """
        default_template = "{stat} = {val:.2g}; p = {p:.2g}"

        # Call the function and determine the form of the return value(s)
        out = func(self.x, self.y)
        try:
            val, p = out
        except TypeError:
            val, p = out, None
            default_template, _ = default_template.split(";")

        # Set the default template
        if template is None:
            template = default_template

        # Default to name of the function
        if stat is None:
            stat = func.__name__

        # Format the annotation
        if p is None:
            annotation = template.format(stat=stat, val=val)
        else:
            annotation = template.format(stat=stat, val=val, p=p)

        # Draw an invisible plot and use the legend to draw the annotation
        # This is a bit of a hack, but `loc=best` works nicely and is not
        # easily abstracted.
        phantom, = self.ax_joint.plot(self.x, self.y, linestyle="", alpha=0)
        self.ax_joint.legend([phantom], [annotation], loc=loc, **kwargs)
        phantom.remove()

        return self

    def set_axis_labels(self, xlabel="", ylabel="", **kwargs):
        """Set the axis labels on the bivariate axes.

        Parameters
        ----------
        xlabel, ylabel : strings
            Label names for the x and y variables.
        kwargs : key, value mappings
            Other keyword arguments are passed to the set_xlabel or
            set_ylabel.

        Returns
        -------
        self : JointGrid instance
            returns `self`

        """
        self.ax_joint.set_xlabel(xlabel, **kwargs)
        self.ax_joint.set_ylabel(ylabel, **kwargs)
        return self

    def savefig(self, *args, **kwargs):
        """Wrap figure.savefig defaulting to tight bounding box."""
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(*args, **kwargs)


def pairplot(data, hue=None, hue_order=None, palette=None,
             vars=None, x_vars=None, y_vars=None,
             kind="scatter", diag_kind="hist", markers=None,
             size=2.5, aspect=1, dropna=True,
             plot_kws=None, diag_kws=None, grid_kws=None):
    """Plot pairwise relationships in a dataset.

    By default, this function will create a grid of Axes such that each
    variable in ``data`` will by shared in the y-axis across a single row and
    in the x-axis across a single column. The diagonal Axes are treated
    differently, drawing a plot to show the univariate distribution of the data
    for the variable in that column.

    It is also possible to show a subset of variables or plot different
    variables on the rows and columns.

    This is a high-level interface for :class:`PairGrid` that is intended to
    make it easy to draw a few common styles. You should use :class:`PairGrid`
    directly if you need more flexibility.

    Parameters
    ----------
    data : DataFrame
        Tidy (long-form) dataframe where each column is a variable and
        each row is an observation.
    hue : string (variable name), optional
        Variable in ``data`` to map plot aspects to different colors.
    hue_order : list of strings
        Order for the levels of the hue variable in the palette
    palette : dict or seaborn color palette
        Set of colors for mapping the ``hue`` variable. If a dict, keys
        should be values  in the ``hue`` variable.
    vars : list of variable names, optional
        Variables within ``data`` to use, otherwise use every column with
        a numeric datatype.
    {x, y}_vars : lists of variable names, optional
        Variables within ``data`` to use separately for the rows and
        columns of the figure; i.e. to make a non-square plot.
    kind : {'scatter', 'reg'}, optional
        Kind of plot for the non-identity relationships.
    diag_kind : {'hist', 'kde'}, optional
        Kind of plot for the diagonal subplots.
    markers : single matplotlib marker code or list, optional
        Either the marker to use for all datapoints or a list of markers with
        a length the same as the number of levels in the hue variable so that
        differently colored points will also have different scatterplot
        markers.
    size : scalar, optional
        Height (in inches) of each facet.
    aspect : scalar, optional
        Aspect * size gives the width (in inches) of each facet.
    dropna : boolean, optional
        Drop missing values from the data before plotting.
    {plot, diag, grid}_kws : dicts, optional
        Dictionaries of keyword arguments.

    Returns
    -------
    grid : PairGrid
        Returns the underlying ``PairGrid`` instance for further tweaking.

    See Also
    --------
    PairGrid : Subplot grid for more flexible plotting of pairwise
               relationships.

    Examples
    --------

    Draw scatterplots for joint relationships and histograms for univariate
    distributions:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns; sns.set(style="ticks", color_codes=True)
        >>> iris = sns.load_dataset("iris")
        >>> g = sns.pairplot(iris)

    Show different levels of a categorical variable by the color of plot
    elements:

    .. plot::
        :context: close-figs

        >>> g = sns.pairplot(iris, hue="species")

    Use a different color palette:

    .. plot::
        :context: close-figs

        >>> g = sns.pairplot(iris, hue="species", palette="husl")

    Use different markers for each level of the hue variable:

    .. plot::
        :context: close-figs

        >>> g = sns.pairplot(iris, hue="species", markers=["o", "s", "D"])

    Plot a subset of variables:

    .. plot::
        :context: close-figs

        >>> g = sns.pairplot(iris, vars=["sepal_width", "sepal_length"])

    Draw larger plots:

    .. plot::
        :context: close-figs

        >>> g = sns.pairplot(iris, size=3,
        ...                  vars=["sepal_width", "sepal_length"])

    Plot different variables in the rows and columns:

    .. plot::
        :context: close-figs

        >>> g = sns.pairplot(iris,
        ...                  x_vars=["sepal_width", "sepal_length"],
        ...                  y_vars=["petal_width", "petal_length"])

    Use kernel density estimates for univariate plots:

    .. plot::
        :context: close-figs

        >>> g = sns.pairplot(iris, diag_kind="kde")

    Fit linear regression models to the scatter plots:

    .. plot::
        :context: close-figs

        >>> g = sns.pairplot(iris, kind="reg")

    Pass keyword arguments down to the underlying functions (it may be easier
    to use :class:`PairGrid` directly):

    .. plot::
        :context: close-figs

        >>> g = sns.pairplot(iris, diag_kind="kde", markers="+",
        ...                  plot_kws=dict(s=50, edgecolor="b", linewidth=1),
        ...                  diag_kws=dict(shade=True))

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "'data' must be pandas DataFrame object, not: {typefound}".format(
                typefound=type(data)))

    if plot_kws is None:
        plot_kws = {}
    if diag_kws is None:
        diag_kws = {}
    if grid_kws is None:
        grid_kws = {}

    # Set up the PairGrid
    diag_sharey = diag_kind == "hist"
    grid = PairGrid(data, vars=vars, x_vars=x_vars, y_vars=y_vars, hue=hue,
                    hue_order=hue_order, palette=palette,
                    diag_sharey=diag_sharey,
                    size=size, aspect=aspect, dropna=dropna, **grid_kws)

    # Add the markers here as PairGrid has figured out how many levels of the
    # hue variable are needed and we don't want to duplicate that process
    if markers is not None:
        if grid.hue_names is None:
            n_markers = 1
        else:
            n_markers = len(grid.hue_names)
        if not isinstance(markers, list):
            markers = [markers] * n_markers
        if len(markers) != n_markers:
            raise ValueError(("markers must be a singleton or a list of "
                              "markers for each level of the hue variable"))
        grid.hue_kws = {"marker": markers}

    # Maybe plot on the diagonal
    if grid.square_grid:
        if diag_kind == "hist":
            grid.map_diag(plt.hist, **diag_kws)
        elif diag_kind == "kde":
            diag_kws["legend"] = False
            grid.map_diag(kdeplot, **diag_kws)

    # Maybe plot on the off-diagonals
    if grid.square_grid and diag_kind is not None:
        plotter = grid.map_offdiag
    else:
        plotter = grid.map

    if kind == "scatter":
        plot_kws.setdefault("edgecolor", "white")
        plotter(plt.scatter, **plot_kws)
    elif kind == "reg":
        from .regression import regplot  # Avoid circular import
        plotter(regplot, **plot_kws)

    # Add a legend
    if hue is not None:
        grid.add_legend()

    return grid


def jointplot(x, y, data=None, kind="scatter", stat_func=stats.pearsonr,
              color=None, size=6, ratio=5, space=.2,
              dropna=True, xlim=None, ylim=None,
              joint_kws=None, marginal_kws=None, annot_kws=None, **kwargs):
    """Draw a plot of two variables with bivariate and univariate graphs.

    This function provides a convenient interface to the :class:`JointGrid`
    class, with several canned plot kinds. This is intended to be a fairly
    lightweight wrapper; if you need more flexibility, you should use
    :class:`JointGrid` directly.

    Parameters
    ----------
    x, y : strings or vectors
        Data or names of variables in ``data``.
    data : DataFrame, optional
        DataFrame when ``x`` and ``y`` are variable names.
    kind : { "scatter" | "reg" | "resid" | "kde" | "hex" }, optional
        Kind of plot to draw.
    stat_func : callable or None, optional
        Function used to calculate a statistic about the relationship and
        annotate the plot. Should map `x` and `y` either to a single value
        or to a (value, p) tuple. Set to ``None`` if you don't want to
        annotate the plot.
    color : matplotlib color, optional
        Color used for the plot elements.
    size : numeric, optional
        Size of the figure (it will be square).
    ratio : numeric, optional
        Ratio of joint axes size to marginal axes height.
    space : numeric, optional
        Space between the joint and marginal axes
    dropna : bool, optional
        If True, remove observations that are missing from ``x`` and ``y``.
    {x, y}lim : two-tuples, optional
        Axis limits to set before plotting.
    {joint, marginal, annot}_kws : dicts, optional
        Additional keyword arguments for the plot components.
    kwargs : key, value pairings
        Additional keyword arguments are passed to the function used to
        draw the plot on the joint Axes, superseding items in the
        ``joint_kws`` dictionary.

    Returns
    -------
    grid : :class:`JointGrid`
        :class:`JointGrid` object with the plot on it.

    See Also
    --------
    JointGrid : The Grid class used for drawing this plot. Use it directly if
                you need more flexibility.

    Examples
    --------

    Draw a scatterplot with marginal histograms:

    .. plot::
        :context: close-figs

        >>> import numpy as np, pandas as pd; np.random.seed(0)
        >>> import seaborn as sns; sns.set(style="white", color_codes=True)
        >>> tips = sns.load_dataset("tips")
        >>> g = sns.jointplot(x="total_bill", y="tip", data=tips)

    Add regression and kernel density fits:

    .. plot::
        :context: close-figs

        >>> g = sns.jointplot("total_bill", "tip", data=tips, kind="reg")

    Replace the scatterplot with a joint histogram using hexagonal bins:

    .. plot::
        :context: close-figs

        >>> g = sns.jointplot("total_bill", "tip", data=tips, kind="hex")

    Replace the scatterplots and histograms with density estimates and align
    the marginal Axes tightly with the joint Axes:

    .. plot::
        :context: close-figs

        >>> iris = sns.load_dataset("iris")
        >>> g = sns.jointplot("sepal_width", "petal_length", data=iris,
        ...                   kind="kde", space=0, color="g")

    Use a different statistic for the annotation:

    .. plot::
        :context: close-figs

        >>> from scipy.stats import spearmanr
        >>> g = sns.jointplot("size", "total_bill", data=tips,
        ...                   stat_func=spearmanr, color="m")

    Draw a scatterplot, then add a joint density estimate:

    .. plot::
        :context: close-figs

        >>> g = (sns.jointplot("sepal_length", "sepal_width",
        ...                    data=iris, color="k")
        ...         .plot_joint(sns.kdeplot, zorder=0, n_levels=6))

    Pass vectors in directly without using Pandas, then name the axes:

    .. plot::
        :context: close-figs

        >>> x, y = np.random.randn(2, 300)
        >>> g = (sns.jointplot(x, y, kind="hex", stat_func=None)
        ...         .set_axis_labels("x", "y"))

    Draw a smaller figure with more space devoted to the marginal plots:

    .. plot::
        :context: close-figs

        >>> g = sns.jointplot("total_bill", "tip", data=tips,
        ...                   size=5, ratio=3, color="g")

    Pass keyword arguments down to the underlying plots:

    .. plot::
        :context: close-figs

        >>> g = sns.jointplot("petal_length", "sepal_length", data=iris,
        ...                   marginal_kws=dict(bins=15, rug=True),
        ...                   annot_kws=dict(stat="r"),
        ...                   s=40, edgecolor="w", linewidth=1)

    """
    # Set up empty default kwarg dicts
    if joint_kws is None:
        joint_kws = {}
    joint_kws.update(kwargs)
    if marginal_kws is None:
        marginal_kws = {}
    if annot_kws is None:
        annot_kws = {}

    # Make a colormap based off the plot color
    if color is None:
        color = color_palette()[0]
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [utils.set_hls_values(color_rgb, l=l)
              for l in np.linspace(1, 0, 12)]
    cmap = blend_palette(colors, as_cmap=True)

    # Initialize the JointGrid object
    grid = JointGrid(x, y, data, dropna=dropna,
                     size=size, ratio=ratio, space=space,
                     xlim=xlim, ylim=ylim)

    # Plot the data using the grid
    if kind == "scatter":

        joint_kws.setdefault("color", color)
        grid.plot_joint(plt.scatter, **joint_kws)

        marginal_kws.setdefault("kde", False)
        marginal_kws.setdefault("color", color)
        grid.plot_marginals(distplot, **marginal_kws)

    elif kind.startswith("hex"):

        x_bins = min(_freedman_diaconis_bins(grid.x), 50)
        y_bins = min(_freedman_diaconis_bins(grid.y), 50)
        gridsize = int(np.mean([x_bins, y_bins]))

        joint_kws.setdefault("gridsize", gridsize)
        joint_kws.setdefault("cmap", cmap)
        grid.plot_joint(plt.hexbin, **joint_kws)

        marginal_kws.setdefault("kde", False)
        marginal_kws.setdefault("color", color)
        grid.plot_marginals(distplot, **marginal_kws)

    elif kind.startswith("kde"):

        joint_kws.setdefault("shade", True)
        joint_kws.setdefault("cmap", cmap)
        grid.plot_joint(kdeplot, **joint_kws)

        marginal_kws.setdefault("shade", True)
        marginal_kws.setdefault("color", color)
        grid.plot_marginals(kdeplot, **marginal_kws)

    elif kind.startswith("reg"):

        from .regression import regplot

        marginal_kws.setdefault("color", color)
        grid.plot_marginals(distplot, **marginal_kws)

        joint_kws.setdefault("color", color)
        grid.plot_joint(regplot, **joint_kws)

    elif kind.startswith("resid"):

        from .regression import residplot

        joint_kws.setdefault("color", color)
        grid.plot_joint(residplot, **joint_kws)

        x, y = grid.ax_joint.collections[0].get_offsets().T
        marginal_kws.setdefault("color", color)
        marginal_kws.setdefault("kde", False)
        distplot(x, ax=grid.ax_marg_x, **marginal_kws)
        distplot(y, vertical=True, fit=stats.norm, ax=grid.ax_marg_y,
                 **marginal_kws)
        stat_func = None
    else:
        msg = "kind must be either 'scatter', 'reg', 'resid', 'kde', or 'hex'"
        raise ValueError(msg)

    if stat_func is not None:
        grid.annotate(stat_func, **annot_kws)

    return grid
