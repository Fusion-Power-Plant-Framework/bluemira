import pandas as pd


def get_percent_err(row):

    # Adds a column to an OpenMC results dataframe that is the
    # percentage stochastic uncertainty in the result
    
    val = 100

    if (isinstance(row["mean"], pd.Series) and row["mean"].any() > 0.0) or row[
        "mean"
    ] > 0.0:
        # For the mesh values the values are in a series but all results are equivalent
        val *= row["std. dev."] / row["mean"]

    return val
