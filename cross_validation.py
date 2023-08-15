#Estes testes são feitos com validação cruzada k-fold, com um k = 5

#puxar cortes de um otimizador, neste caso, ecolhi o diff-evol
import pandas as pd
import numpy as np
import cut_the_tails as ct
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold

'''

    Modelo do dataframe a ser criado por este script:

    Modelo|Base|Otimizador|Replica|Inf_cut|Sup_cut|MAPE

    Modelo: Modelo ML e Classificador utilizados para o treinamento/testes
    Base: A Base de dados 
    Otimizador: Otimizador utilizado para definir os cortes
    Replica: Réplicas para testes idênticos
    Inf_cut: Corte da cauda inferior
    Sup_cut: Corte da cauda superior
    MAPE: MAPE adquirido pelos testes, utilizando os cortes definidos pelo otimizador

    No final da execução deste script, será gerado um arquivo csv de nome "optimal_tails.csv" no mesmo nível que este script.

'''

Models = ['RandomForest']

Bases = ['bike_sharing_hour']
#, 'Blueberry_Yield', 'car_price', 'employee_performance', 'house_rent', 'medical_cost', 'solar_radiation', 'Financial_Distress', 'Real_estate', 'wind_dataset'

Otimizadores = ['differential-evol'] 

Replicas = [1]

#criando um dataframe para o output
df_output = pd.DataFrame(columns=['Modelo', 'Base', 'Otimizador', 'Replica', 'Inf_cut', 'Sup_cut', 'MAPE_test'])

# K = 5 para a validação cruzada
n_split = 5

kf = KFold(n_splits=n_split)

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
    
    if name == 'Blueberry_Yield':
        target = 'yield'
        features = ['clonesize',
        'honeybee',
        'bumbles',
        'andrena',
        'osmia',
        'MaxOfUpperTRange',
        'MinOfUpperTRange',
        'AverageOfUpperTRange',
        'MaxOfLowerTRange',
        'MinOfLowerTRange',
        'AverageOfLowerTRange',
        'RainingDays',
        'AverageRainingDays',
        'fruitset',
        'fruitmass',
        'seeds']
    
    if name == 'car_price':
        target = 'remainder__selling_price'
        features = ['onehotencoder__fuel_CNG',
        'onehotencoder__fuel_Diesel',
        'onehotencoder__fuel_LPG',
        'onehotencoder__fuel_Petrol',
        'onehotencoder__seller_type_Dealer',
        'onehotencoder__seller_type_Individual',
        'onehotencoder__seller_type_Trustmark Dealer',
        'onehotencoder__transmission_Automatic',
        'onehotencoder__transmission_Manual',
        'onehotencoder__owner_First Owner',
        'onehotencoder__owner_Fourth & Above Owner',
        'onehotencoder__owner_Second Owner',
        'onehotencoder__owner_Test Drive Car',
        'onehotencoder__owner_Third Owner',
        'remainder__year',
        'remainder__km_driven',
        'remainder__mileage',
        'remainder__engine',
        'remainder__max_power',
        'remainder__seats']
    
    if name == 'employee_performance':
        target = 'actual_productivity'
        features = ['team',
        'targeted_productivity',
        'smv',
        'wip',
        'over_time',
        'incentive',
        'idle_time',
        'idle_men',
        'no_of_style_change',
        'no_of_workers',
        'month',
        'quarter_Quarter1',
        'quarter_Quarter2',
        'quarter_Quarter3',
        'quarter_Quarter4',
        'quarter_Quarter5',
        'department_finishing']
    
    if name == 'house_rent':
        target = 'remainder__Rent'
        features = ['onehotencoder__City_Bangalore',
        'onehotencoder__City_Chennai',
        'onehotencoder__City_Delhi',
        'onehotencoder__City_Hyderabad',
        'onehotencoder__City_Kolkata',
        'onehotencoder__City_Mumbai',
        'onehotencoder__Area Type_Built Area',
        'onehotencoder__Area Type_Carpet Area',
        'onehotencoder__Area Type_Super Area',
        'onehotencoder__Furnishing Status_Furnished',
        'onehotencoder__Furnishing Status_Semi-Furnished',
        'onehotencoder__Furnishing Status_Unfurnished',
        'onehotencoder__Tenant Preferred_Bachelors',
        'onehotencoder__Tenant Preferred_Bachelors/Family',
        'onehotencoder__Tenant Preferred_Family',
        'onehotencoder__Point of Contact_Contact Agent',
        'onehotencoder__Point of Contact_Contact Builder',
        'onehotencoder__Point of Contact_Contact Owner',
        'remainder__BHK',
        'remainder__Size',
        'remainder__Bathroom']
    
    if name == 'medical_cost':
        target = 'remainder__charges'
        features = ['onehotencoder__sex_female',
        'onehotencoder__sex_male',
        'onehotencoder__smoker_no',
        'onehotencoder__smoker_yes',
        'onehotencoder__region_northeast',
        'onehotencoder__region_northwest',
        'onehotencoder__region_southeast',
        'onehotencoder__region_southwest',
        'remainder__age',
        'remainder__bmi',
        'remainder__children']
    
    if name == 'solar_radiation':
        target = 'Radiation'
        features = [
        'Temperature',
        'Pressure',
        'Humidity',
        'WindDirection(Degrees)',
        'Speed',
        'Time',
        'TimeSunRise',
        'TimeSunSet',
        ]
    
    if name == 'wind_dataset':
        target = 'WIND'
        features = [
        'IND',
        'RAIN',
        'IND.1',
        'T.MAX',
        'IND.2',
        'T.MIN',
        'T.MIN.G'
        ]
    
    if name == 'Financial_Distress':
        target = 'Financial Distress'
        features = ['Company',
        'Time', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 
        'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 
        'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 
        'x41', 'x42', 'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49', 'x50', 
        'x51', 'x52', 'x53', 'x54', 'x55', 'x56', 'x57', 'x58', 'x59', 'x60', 
        'x61', 'x62', 'x63', 'x64', 'x65', 'x66', 'x67', 'x68', 'x69', 'x70', 
        'x71', 'x72', 'x73', 'x74', 'x75', 'x76', 'x77', 'x78', 'x79', 'x80', 
        'x81', 'x82', 'x83']
    
    if name == 'Real_estate':
        target = 'Y house price of unit area'
        features = ['X1 transaction date',
        'X2 house age',
        'X3 distance to the nearest MRT station',
        'X4 number of convenience stores',
        'X5 latitude',
        'X6 longitude']
    
    return target, features

def select_model_classifier(name):
    if name == 'RandomForest':
        model = RandomForestRegressor(max_depth=5, random_state=0) 
        classifier = RandomForestClassifier(max_depth=5, random_state=0)
    
    if name == 'DecisionTree':
        model = DecisionTreeRegressor(random_state=0, max_depth=5) 
        classifier = DecisionTreeClassifier(random_state=0, max_depth=5)

    return model, classifier

def select_optimal_cuts(name):
    if name == 'bike_sharing_hour':
        return [0.05306370216848327,0.45386215659447826]
    
    if name == 'Blueberry_Yield':
        return [0.4862264584551263,0.9096903279899677]
    
    if name == 'car_price':
        return [0.5240341528945671,0.9422140879397973]
    
    if name == 'employee_performance':
        return [0.20869186955399333,0.8903139210638581]
    
    if name == 'house_rent':
        return [0.03616334850717995,0.24984895071529156]
    
    if name == 'medical_cost':
        return [0.177524584292729,0.692002190352954]
    
    if name == 'solar_radiation':
        return [0.03820002997434446,0.5527332482675485]
    
    if name == 'wind_dataset':
        return [0.04137954538219557,0.15474822607954314]
    
    if name == 'Financial_Distress':
        return [0.12048951955607584,0.20917882224220985]
    
    if name == 'Real_estate':
        return [0.10576030517807855,0.3108907006822149]
    
    return target, features

for i in Models:
    for j in Bases:
        for k in Otimizadores:
            for l in Replicas:
                #----------------executando o otimizador para encontrar os melhores cortes----------------#

                #selecionando modelo e classificador
                model, classifier = select_model_classifier(i)
                
                #carregando o dataframe da base selecionada
                df = pd.read_csv('preprocessed_data_sets/' + j + '.csv')
                df.dropna(inplace=True)
                df = df.reset_index(drop=True)

                #selecionando os features e target do dataframe
                target, features = select_feature_target(j)

                #importando os cortes ótimos
                x = select_optimal_cuts(j)
                
                if x[0] > x[1]:
                    x[1], x[0] = x[0], x[1]

                #partilhamento de treinamento para o KFold/teste final
                X_Kfold, X_final_test, Y_Kfold, Y_final_test = train_test_split(df, df[target], test_size=0.2)
                base_KFold = pd.concat([X_Kfold, Y_Kfold])  #Juntando os dados para o conjunto destinado a validação KFold
                base_teste_final = pd.concat([X_final_test, Y_final_test])  #Juntando os dados para o conjunto destinado ao teste final
                print(base_KFold)

                #utilizando os percentis ótimos encontrados pelo otimizador para partir as caudas
                base_KFold = ct.split_by_quantile_class(base_KFold, target, x)
                print(base_KFold)

                #partilhamento da base de dados para validação cruzada
                #treinamento dos modelos 
                for train_index, test_index in kf.split(base_KFold):
                    train = base_KFold.iloc[train_index]
                    test = base_KFold.iloc[test_index]
                    test.dropna(inplace=True)
                    print(train)
                    X_train = train[features].to_numpy()
                    Y_train = train[target].to_numpy()
                    X_test = test[features].to_numpy()
                    Y_test = test[target].to_numpy()
                    y_tail = train['tail_class'].to_numpy()

                    tail_classifier = ct.fit_tail_classifier(X_train,y_tail,classifier)
                    models = ct.fit_tail_models(X_train,Y_train,y_tail,model)

                    #predicting...
                    y_tail = ct.batch_tail_predict(X_test,tail_classifier,models)

                    Mape = mean_absolute_percentage_error(y_tail,Y_test)

                    print(Mape)


                '''
                _X_train, _X_test, y_train, y_test = train_test_split(cdf, cdf[target], test_size=0.2)
                
                
                
                y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

                #construindo os modelos e classificadores
                tail_classifier = ct.fit_tail_classifier(X_train,y_tail,classifier)


                models = ct.fit_tail_models(X_train,y_train,y_tail,model)

                #------------------------passando os dados para um arquivo CSV------------------------#

                new_row = {'Modelo': i, 'Base': j, 'Otimizador': k, 'Replica': l, 'Inf_cut': x[0], 'Sup_cut': x[1], 'MAPE_test': Mape_test}
                df_output.loc[-1] = new_row
                df_output.index = df_output.index + 1
                df_output = df_output.sort_index()

        #convertendo o dataframe gerado para um CSV
        df_output.to_csv('cross_validation_results.csv', encoding='utf-8', index=False)
        '''