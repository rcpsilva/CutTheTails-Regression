import numpy as np

def split_by_quantile_class(df,target,q):
    '''
    Args

        df: original dataset
        target: target variable
        q: list of upper and lower quantile

    Returns:

        lbdf: data below the lower quantile
        nddf: data between quantiles
        ubdf: date over the upper quantile

    '''

    q = np.quantile(df[[target]],q = q)

    quant_class = []

    for index, row in df.iterrows():
        if row[target] <= q[0]:
            quant_class.append(1)
        elif (row[target] > q[0]) and (row[target] < q[1]):
            quant_class.append(2)
        elif row[target] >= q[1]:
            quant_class.append(3)
    
    df['quant_class'] = quant_class

    return df

def split_by_quantile(df,target,q):
    '''
    Args

        df: original dataset
        target: target variable
        q: list of upper and lower quantile

    Returns:

        lbdf: data below the lower quantile
        nddf: data between quantiles
        ubdf: date over the upper quantile

    '''

    q = np.quantile(df[[target]],q = q)

    lbdf = df[df[target] <= q[0]]
    ubdf = df[df[target] >= q[1]]
    nddf = df[(df[target] > q[0]) & (df[target] < q[1])]

    return lbdf,nddf,ubdf