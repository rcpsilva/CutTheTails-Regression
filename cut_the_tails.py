import numpy as np
from copy import deepcopy

def fit_cut_the_tail_proxy(X,y,quantiles,IQR,proxy_model,lower_tail_model,normal_model,upper_tail_model):

    proxy_model.fit(X,y)

    lower_outlier = (y < (quantiles[0] - 1.5*IQR))
    upper_outlier = (y > (quantiles[1] + 1.5*IQR))
    normal = (y >= (quantiles[0] - 1.5*IQR)) & (y <= (quantiles[1] + 1.5*IQR))

    if np.sum(lower_outlier) > 1:
        lower_tail_model.fit(X[lower_outlier],y[lower_outlier])
    else:
        lower_tail_model.fit(X[normal],y[normal])

    if np.sum(upper_outlier) > 1:
        upper_tail_model.fit(X[upper_outlier],y[upper_outlier])
    else:
        upper_tail_model.fit(X[normal],y[normal])

    normal_model.fit(X[normal],y[normal])

def predict_cut_the_tail_proxy(X,quantiles,IQR,proxy_model,lower_tail_model,normal_model,upper_tail_model):

    y_proxy = proxy_model.predict(X)

    lower_outlier = (y_proxy < (quantiles[0] - 1.5*IQR))
    upper_outlier = (y_proxy > (quantiles[1] + 1.5*IQR))
    normal = (y_proxy >= (quantiles[0] - 1.5*IQR)) & (y_proxy <= (quantiles[1] + 1.5*IQR))

    y_lower = lower_tail_model.predict(X)
    y_normal = normal_model.predict(X)
    y_upper = upper_tail_model.predict(X)

    return y_lower*lower_outlier + y_normal*normal + y_upper*upper_outlier


def fit_cut_the_tail(X,y,quantiles,tail_classifier,lower_tail_model,normal_model,upper_tail_model):
    """This function takes in the features X and the target variable y, 
    as well as some parameters to specify the quantiles for defining the tails of the distribution
    and the models to fit for each tail.

    Args:
        x: A numpy array of features.
        y: A numpy array of the target variable.
        quantiles: A list of two values specifying the quantiles to use for defining the tails of the distribution.
        tail_classifier: A machine learning model to fit for assigning observations to each tail class.
        lower_tail_model: A machine learning model to fit for the lower tail.
        normal_model: A machine learning model to fit for the normal range.
        upper_tail_model: A machine learning model to fit for the upper tail.
    
    Returns:
        None
    
    """


    # Calculate the quantiles of the target variable y to define the lower and upper tails
    q = np.quantile(y,q = quantiles)

    # Convert the target variable y to a numpy array
    y = np.array(y)

    # Define tail classes for each observation based on the calculated quantiles
    # Assign 1 for lower tail, 2 for normal, and 3 for upper tail
    y_tail = (y <= q[0])*1 + (y >= q[1])*3 + ((y > q[0]) & (y < q[1]))*2

    # Fit the tail classifier using the features X and the assigned tail classes
    tail_classifier.fit(X,y_tail)

    # Fit the lower tail model using the subset of X and y where y_tail == 1
    lower_tail_model.fit(X[y_tail==1],y[y_tail==1])

    # Fit the normal model using the subset of X and y where y_tail == 2
    normal_model.fit(X[y_tail==2],y[y_tail==2])

    # Fit the upper tail model using the subset of X and y where y_tail == 3
    upper_tail_model.fit(X[y_tail==3],y[y_tail==3])



def predict_cut_the_tails(x, tail_classifier, lower_tail_model, normal_model, upper_tail_model):
    """
    Predicts the target variable based on a set of features x, using different models for the lower, normal, and upper
    tails of the distribution based on a tail classifier.

    Args:
        x: A numpy array of features.
        tail_classifier: A trained machine learning model to classify observations into tail classes.
        lower_tail_model: A trained machine learning model for the lower tail.
        normal_model: A trained machine learning model for the normal range.
        upper_tail_model: A trained machine learning model for the upper tail.

    Returns:
        A numpy array of predicted values for the target variable.
    """

    # Use the tail classifier to predict the tail class for each observation in x
    y_tail = tail_classifier.predict(x)

    # Use the corresponding model to predict the target variable for each observation in x,
    # based on its assigned tail class
    y_lower = lower_tail_model.predict(x[y_tail == 1])
    y_normal = normal_model.predict(x[y_tail == 2])
    y_upper = upper_tail_model.predict(x[y_tail == 3])

    # Combine the predicted values for each observation based on its assigned tail class
    y = y_lower * (y_tail == 1) + y_normal * (y_tail == 2) + y_upper * (y_tail == 3)

    return y


def fit_tail_classifier(X,y_tail,model):
    model.fit(X,y_tail)
    return model

def fit_tail_models(X,y,y_tail,model):
    models = [deepcopy(model) for i in [0,1,2]]
    for i in [0,1,2]:
        models[i].fit(X[y_tail==i],y[y_tail==i]) 
    return models

def tail_predict(x,tail_classifier,tail_models):
    tail_class = tail_classifier.predict([x])
    return tail_models[tail_class[0]].predict([x])

def batch_tail_predict(X,tail_classifier,tail_models):
    t_class = tail_classifier.predict(X)
    
    preds = np.zeros((len(t_class),1))
    for i in range(len(t_class)): #very inneficient
        preds[i] = tail_models[t_class[i]].predict([X[i]])[0]
    
    return preds

def split_by_quantile_class(df,target,q):
    ''' Creates a column called 'quant_class' that identifies the samples in the 
        target distribution tails. Lower tail:0, normal:1, upper tail:2
    Args

        df: original dataset
        target: target variable
        q: list of upper and lower quantile

    Returns:

        df: updated dataframe

    '''

    q = np.quantile(df[[target]],q = q)

    quant_class = []

    for index, row in df.iterrows():
        if row[target] <= q[0]:
            quant_class.append(0)
        elif (row[target] > q[0]) and (row[target] < q[1]):
            quant_class.append(1)
        elif row[target] >= q[1]:
            quant_class.append(2)
    
    df['tail_class'] = quant_class

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

def get_optimal_percentiles_brute_force(X,y,tail_classifier,lower_tail_model,normal_model,upper_tail_model):

    percentiles = [0,1]

    # TO_DO

    return percentiles

def get_optimal_percentiles_nelder_mead(X,y,tail_classifier,lower_tail_model,normal_model,upper_tail_model):

    percentiles = [0,1]

    # TO_DO

    return percentiles

def objective(x,args):

    # TO_DO

    return x

