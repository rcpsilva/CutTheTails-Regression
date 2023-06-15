import pandas as pd
import numpy as np
from scipy.stats import moment, kurtosis, skew

Bases = ['bike_sharing_hour', 'Blueberry_Yield', 'car_price', 'employee_performance', 'house_rent', 'medical_cost', 'solar_radiation', 'Financial_Distress', 'Real_estate', 'wind_dataset']

Orders = [1, 2, 3, 4, 5, 6]

#criando um dataframe para o output
df_output = pd.DataFrame(columns=['Base', 'ordem_1', 'ordem_2', 'ordem_3', 'ordem_4', 'ordem_5', 'ordem_6', 'kurtosis', 'skewness'])

def select_feature_target(name):
    if name == 'bike_sharing_hour':
        target = 'cnt'
    
    if name == 'Blueberry_Yield':
        target = 'yield'
    
    if name == 'car_price':
        target = 'remainder__selling_price'
    
    if name == 'employee_performance':
        target = 'actual_productivity'
    
    if name == 'house_rent':
        target = 'remainder__Rent'
    
    if name == 'medical_cost':
        target = 'remainder__charges'
    
    if name == 'solar_radiation':
        target = 'Radiation'
    
    if name == 'wind_dataset':
        target = 'WIND'
    
    if name == 'Financial_Distress':
        target = 'Financial Distress'
    
    if name == 'Real_estate':
        target = 'Y house price of unit area'
    
    return target

for i in Bases:

    #carregando o dataframe da base selecionada
    df = pd.read_csv('preprocessed_data_sets\\' + i + '.csv')
    
    target = select_feature_target(i)

    order1 = moment(df[target], moment=1) #deve ser 0.0, pois o momento centralizado de primeira ordem de qualquer distribuição é zero
    order2 = moment(df[target], moment=2) 
    order3 = moment(df[target], moment=3) 
    order4 = moment(df[target], moment=4) 
    order5 = moment(df[target], moment=5) 
    order6 = moment(df[target], moment=6) 

    #Em Kurtose, quanto maior o resultado, mais pesada é a cauda em comparação com a distribuição normal, leptokurtic
    kurt = kurtosis(df[target], fisher= True)

    #Em Skewness, quanto maior o resultado, sigifica que a cauda direita é mais pesada.
    skewness = skew(df[target])

    new_row = {'Base': i, 'ordem_1': order1, 'ordem_2': order2, 'ordem_3': order3, 'ordem_4': order4, 'ordem_5': order5, 'ordem_6': order6, 'kurtosis': kurt, 'skewness': skewness}
    df_output.loc[-1] = new_row
    df_output.index = df_output.index + 1
    df_output = df_output.sort_index()

    df_output.to_csv('data_moments.csv', encoding='utf-8', index=False)
