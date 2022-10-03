import pandas as pd


def get_percent_err(row):
    
    # Adds a column to an OpenMC results dataframe that is the 
    # percentage stochastic uncertainty in the result

    if isinstance( row['mean'], pd.Series ):  
        # For the mesh values the values are in a series but all results are equivalent
        
        if row['mean'].any() > 0.0:
            val = 100. * row['std. dev.'] / row['mean'] 
        else:
            val = 100.      
    else:
        if row['mean'] > 0.0:
            val = 100. * row['std. dev.'] / row['mean'] 
        else:
            val = 100.
            
    return val
