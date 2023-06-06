import pandas as pd
import numpy as np
import cut_the_tails as ct
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

'''

    Modelo do dataframe a ser criado por este script:

    Modelo|Base|Otimizador|Replica|Inf_cut|Sup_cut|MAPE_otimizador|MAPE_test

    Modelo: Modelo ML e Classificador utilizados para o treinamento/testes
    Base: A Base de dados 
    Otimizador: Otimizador utilizado para definir os cortes
    Replica: Réplicas para testes idênticos
    Inf_cut: Corte da cauda inferior
    Sup_cut: Corte da cauda superior
    MAPE_otimizador: MAPE adquirido pelo otimizador
    MAPE_test: MAPE adquirido pelos testes, utilizando os cortes definidos pelo otimizador

    No final da execução deste script, será gerado um arquivo csv de nome "optimal_tails.csv" no mesmo nível que este script.

'''

Models = ['RandomForest']

Bases = ['bike_sharing_hour']

Otimizadores = ['brute']
#, 'direct', 'differential-evol'

Replicas = [1]

df = pd.DataFrame(columns=['Modelo', 'Base', 'Otimizador', 'Replica', 'Inf_cut', 'Sup_cut', 'MAPE_otimizador', 'MAPE_test'])

def select_feature_target(name):
    if name == 'bike_sharing_hour':
        target = 'cnt'
        features = ['season',
        'mnth',
        'hr',
        'holiday',
        'weekday',
        'workingday',
        'weathersit',
        'temp',
        'atemp',
        'hum',
        'windspeed']
    
    return target, features

def select_model_classifier(name):
    if i == name:
        model = RandomForestRegressor(max_depth=5) 
        classifier = RandomForestClassifier(max_depth=5)
    
    return model, classifier

for i in Models:
    for j in Bases:
        for k in Otimizadores:
            for l in Replicas:
                #----------------executando o otimizador para encontrar os melhores cortes----------------#

                #selecionando modelo e classificador
                model, classifier = select_model_classifier(i)
                
                #carregando o dataframe da base selecionada
                df = pd.read_csv('data_sets\\' + j + '.csv')

                #selecionando os features e target do dataframe
                target, features = select_feature_target(j)

                #calculando os cortes ótimos
                x, fval = ct.get_cuts_direct_optimization(df, target, features, classifier, model, k)

                #------------------------testando os cortes feitos pelo otimizador------------------------#

                if x[0] > x[1]:
                    x[1], x[0] = x[0], x[1]
                
                #utilizando os percentis ótimos encontrados pelo otimizador
                cdf = ct.split_by_quantile_class(df, target, x)

                #partilhamento de treinamento/teste
                _X_train, _X_test, y_train, y_test = train_test_split(cdf, cdf[target], test_size=0.2)
                X_train = _X_train[features].to_numpy()
                X_test = _X_test[features].to_numpy()
                y_tail = _X_train['tail_class'].to_numpy()
                y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

                #construindo os modelos e classificadores
                tail_classifier = ct.fit_tail_classifier(X_train,y_tail,classifier)
                models = ct.fit_tail_models(X_train,y_train,y_tail,model)

                #predicting...
                y_tail = ct.batch_tail_predict(X_test,tail_classifier,models)

                Mape_test = mean_absolute_percentage_error(y_tail,y_test)

                #------------------------passando os dados para um arquivo CSV------------------------#

                new_row = {'Modelo': i, 'Base': j, 'Otimizador': k, 'Replica': l, 'Inf_cut': x[0], 'Sup_cut': x[1], 'MAPE_otimizador': fval, 'MAPE_test': Mape_test}
                df.loc[len(df)] = new_row

#convertendo o dataframe gerado para um CSV
df.to_csv('optimal_tails.csv', encoding='utf-8', index=False)