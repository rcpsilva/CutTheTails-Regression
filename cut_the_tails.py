import numpy as np
from copy import deepcopy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import minimize, basinhopping, Bounds
from numpy.random import rand
import matplotlib.pyplot as plt

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
    q = np.quantile(y,q = quantiles)

    y = np.array(y)

    # get tail classes
    y_tail = (y <= q[0])*1 + (y >= q[1])*3 + ((y > q[0]) & (y < q[1]))*2

    # fit tail classifier

    tail_classifier.fit(X,y_tail)

    # fit lower tail model
    lower_tail_model.fit(X[y_tail==1],y[y_tail==1])

    # fit normal model
    normal_model.fit(X[y_tail==2],y[y_tail==2])

    # fit upper tail model
    upper_tail_model.fit(X[y_tail==3],y[y_tail==3])
    

def predict_cut_the_tails(X,tail_classifier,lower_tail_model,normal_model,upper_tail_model):

    y_tail = tail_classifier.predict(X)

    y_lower = lower_tail_model.predict(X)
    y_normal = normal_model.predict(X)
    y_upper = upper_tail_model.predict(X)

    y = y_lower*(y_tail==1) + y_normal*(y_tail==2) + y_upper*(y_tail==3)

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
    
    df['tail_class'] = pd.Series(quant_class)

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

def get_mode_histogram(df, target, bins):
    '''
        Args

            df: original dataset
            target: target variable
            bins: quantity of bins for the histogram

        Returns:

            mode_percentage: the percentage refering to the mode of the distribution, rounded

    '''
    n, bins, patches = plt.hist(df[target], bins=bins)

    mode_index = n.argmax()
    
    mode = round((bins[mode_index] + bins[mode_index+1])/2)

    mode_percentage = round(mode/bins[99], 2)

    return mode_percentage

def objective_one_tail(x, df, target, features, classifier, model, mode):

    #select the percentile and classify the entire dataframe
    cdf = split_by_quantile_class(df, target,[0.0, x])

    X = cdf[features].to_numpy()
    y_tail = cdf['tail_class'].to_numpy()
    y = cdf[target].to_numpy()

    #test/train splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train_aux, X_test_aux, y_train_tail, y_test_tail = train_test_split(X, y_tail, test_size=0.2, random_state=0)

    #building the classifier and ML models
    tail_classifier = fit_tail_classifier(X,y_tail,classifier)
    models = fit_tail_models(X_train,y_train,y_train_tail,model)

    #Predicting
    y_tail = batch_tail_predict(X_test,tail_classifier,models)

    Mape = mean_absolute_percentage_error(y_tail,y_test)

    return Mape

def objective_two_tail(x, y, df, target, features, classifier, model):  

    x[1], x[0] = x[0], x[1] if x[0] > x[1] else x[1], x[0] 


    #select the percentile and classify the entire dataframe
    cdf = split_by_quantile_class(df, target, x)

    X = cdf[features].to_numpy()
    y_tail = cdf['tail_class'].to_numpy()
    y = cdf[target].to_numpy()

    #test/train splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train_aux, X_test_aux, y_train_tail, y_test_tail = train_test_split(X, y_tail, test_size=0.2, random_state=0)

    #building the classifier and ML models
    tail_classifier = fit_tail_classifier(X,y_tail,classifier)
    models = fit_tail_models(X_train,y_train,y_train_tail,model)

    #Predicting
    y_tail = batch_tail_predict(X_test,tail_classifier,models)

    Mape = mean_absolute_percentage_error(y_tail,y_test)

    return Mape

def get_optimal_percentiles_one_tail_brute_force(df, target, features, classifier, model):
    ''' Find the best percentile for the cut in a one tailed distribuition, using brute force.
        This function compares the MAPE from the results of all the possible percentiles starting at 100% and decreasing by 5% each iteration.
        It returns the percentiles with the best MAPE.
    
    Args

        df: original dataset
        target: target variable
        classifier: classifier for the classification of the data for the tests
        model: regression models for the superior tail and the middle

    Returns:

        best_percentiles: the percentiles that achieved the lesser MAPE

    '''

    best_MAPE = 10000.0
    mode = get_mode_histogram(df, target, 100)
    best_percentiles = [0, 100]

    for i in np.arange(1.0, mode , -0.05):
        #select the percentile and classify the entire dataframe
        cdf = split_by_quantile_class(df, target,[0.0, round(i, 2)])

        X = cdf[features].to_numpy()
        y_tail = cdf['tail_class'].to_numpy()
        y = cdf[target].to_numpy()

        #test/train splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train_aux, X_test_aux, y_train_tail, y_test_tail = train_test_split(X, y_tail, test_size=0.2, random_state=0)

        #building the classifier and ML models
        tail_classifier = fit_tail_classifier(X,y_tail,classifier)
        models = fit_tail_models(X_train,y_train,y_train_tail,model)

        #Predicting
        y_tail = batch_tail_predict(X_test,tail_classifier,models)

        #Calculating MAPE and comparing with best result
        current_MAPE = mean_absolute_percentage_error(y_tail,y_test)
        if current_MAPE < best_MAPE:
            best_MAPE = current_MAPE
            best_percentiles = [0.0, round(i, 2)]

    return best_percentiles

def get_optimal_percentiles_two_tail_brute_force(df, target, features, classifier, model):
    ''' 
        Find the best percentile for the cut in a two tailed distribuition, using brute force.
        This function compares the MAPE from the results of all the possible percentiles starting at 100% and decreasing by 5% each iteration.
        It returns the percentiles with the best MAPE.
    
    Args

        df: original dataset
        target: target variable
        classifier: classifier for the classification of the data for the tests
        model: regression models for the superior tail and the middle

    Returns:

        best_percentiles: the percentiles that achieved the lesser MAPE

    '''

    best_MAPE = 10000.0
    mode = get_mode_histogram(df, target, 100)
    best_percentiles = [0, 100]

    for i in np.arange(0, mode, 0.05):
        for j in np.arange(1.0, mode, -0.05):
            #select the percentile and classify the entire dataframe
            cdf = split_by_quantile_class(df, target,[round(i, 2), round(j, 2)])

            X = cdf[features].to_numpy()
            y_tail = cdf['tail_class'].to_numpy()
            y = cdf[target].to_numpy()

            #test/train splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            X_train_aux, X_test_aux, y_train_tail, y_test_tail = train_test_split(X, y_tail, test_size=0.2, random_state=0)

            #building the classifier and ML models
            tail_classifier = fit_tail_classifier(X,y_tail,classifier)
            models = fit_tail_models(X_train,y_train,y_train_tail,model)

            #Predicting
            y_tail = batch_tail_predict(X_test,tail_classifier,models)

            #Calculating MAPE and comparing with best result
            current_MAPE = mean_absolute_percentage_error(y_tail,y_test)
            if current_MAPE < best_MAPE:
                best_MAPE = current_MAPE
                best_percentiles = [round(i, 2), round(j, 2)]

    return best_percentiles

def get_optimal_percentiles_one_tail_nelder_mead(df, target, features, classifier, model, method):

    bnd = Bounds(0.0, 1.0)
    mode = get_mode_histogram(df, target, 100)
    r_min, r_max = 0.0, 1.0
    x0 = r_min + rand(2) * (r_max - r_min)
    
    if method == 'normal':
        result = minimize(objective_one_tail, x0, args = (df, target, features, classifier, model, mode), method="nelder-mead", bounds=bnd)
    if method == 'basin-hopping':
        result = basinhopping(objective_one_tail, x0, minimizer_kwargs={'method':'nelder-mead', 'args': (df, target, features, classifier, model), 'bounds': bnd}, stepsize=5.0, niter=100)

    percentiles = result['x']

    return percentiles

def get_optimal_percentiles_two_tail_nelder_mead(df, target, features, classifier, model, method):
    return 0

def get_optimal_percentiles(df, target, features, classifier, model, method, algorythm, qnt_of_tails):
    #methods are normal and basin hopping
    #algorythms are: brute force, nelder mead...

    if algorythm == 'brute-force':
        if qnt_of_tails == 1:
            percentiles = get_optimal_percentiles_one_tail_brute_force(df, target, features, classifier, model)
        else:
            percentiles = get_optimal_percentiles_two_tail_brute_force(df, target, features, classifier, model)
    elif algorythm == 'nelder-mead':
        if qnt_of_tails == 1:
            percentiles = get_optimal_percentiles_one_tail_nelder_mead(df, target, features, classifier, model, method)
        else:
            percentiles = get_optimal_percentiles_two_tail_nelder_mead(df, target, features, classifier, model, method)
    return percentiles