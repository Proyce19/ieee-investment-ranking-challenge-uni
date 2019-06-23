import base64
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stat
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Dropout
from keras.models import Model
from keras import backend as K
from keras.layers import BatchNormalization
from keras.optimizers import Adam, Adadelta
from sklearn import preprocessing
from sklearn.svm import LinearSVR as SVC
#pd.options.mode.chained_assigment=None

dataset_file_path = "data/full_dataset.csv"
pred_template_file_path = "data/predictions_template.csv"




data = pd.read_csv(dataset_file_path)
pred_template = pd.read_csv(pred_template_file_path)

variable_list = ["X" + str(i) + '_' for i in range(1,71)]



for var in variable_list:
    data[var + 'avg'] = data.filter(regex=(var)).mean(axis = 1)






for var in variable_list:
    data[var + 'avg' + '_pctile'] = stat.rankdata(data[var + 'avg'])/data[var + 'avg'].shape[0]
model_data = pd.concat([data.iloc[:,0:5],data.filter(regex = ('avg'))],axis=1)
time_periods = np.unique(model_data['time_period'],return_index=True)[0]
time_periods_index = np.unique(model_data['time_period'],return_index=True)[1]


def randomForest(train_start_period, prediction_period):
    train_window_start = time_periods_index[time_periods == train_start_period][0]
    train_window_end = time_periods_index[time_periods == prediction_period][0]

    rf_model_data = model_data.iloc[range(train_window_start, train_window_end), :]

    rf_model_data.fillna(0, inplace=True)

    rf = RandomForestRegressor(n_estimators=1, verbose=2, oob_score=True, max_features=10)

    # fit using training data only (Train == 1)
    rf.fit(rf_model_data.loc[rf_model_data['Train'] == 1, 'X1_avg':'X70_avg_pctile'],
           rf_model_data.loc[rf_model_data['Train'] == 1, 'Norm_Ret_F6M'])


    return rf


def sequentalModel(train_start_period,prediction_period):
    train_window_start = time_periods_index[time_periods == train_start_period][0]
    train_window_end = time_periods_index[time_periods == prediction_period][0]

    rf_model_data = model_data.iloc[range(train_window_start, train_window_end), :]

    rf_model_data.fillna(0, inplace=True)

    f=rf_model_data.loc[rf_model_data['Train'] == 1,'X1_avg':'X70_avg_pctile']
    x = f.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    model = Sequential()
    model.add(Dense(256, input_dim=f.shape[1], activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(15, activation='sigmoid'))
    model.add(Dense(10,  activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',optimizer='Adam', metrics=['accuracy'])
    model.fit(f.as_matrix(),rf_model_data.loc[rf_model_data['Train'] == 1,'Norm_Ret_F6M'].as_matrix(),verbose=2,batch_size=10,epochs=10)

    return model


def xgBoost(train_start_period, prediction_period):

    train_window_start = time_periods_index[time_periods == train_start_period][0]
    train_window_end = time_periods_index[time_periods == prediction_period][0]

    xg_model_data = model_data.iloc[range(train_window_start, train_window_end), :]
    xg_model_data.fillna(0,inplace = True)

    dtrain = xgb.DMatrix(data=xg_model_data.loc[xg_model_data['Train'] == 1, 'X1_avg':'X70_avg_pctile'],
                         label=xg_model_data.loc[xg_model_data['Train'] == 1, 'Norm_Ret_F6M'])
    params = {'eta': 0.1, 'seed': 0, 'subsample': 1.0, 'colsample_bytree': 1.0, 'objective': 'reg:linear',
              'max_depth': 6, 'min_child_weight': 1}
    watchlist = [(dtrain, 'train')]
    xg = xgb.train(params, dtrain, 500, watchlist, verbose_eval=20)


    return xg
def light_gbm(train_start_period, prediction_period):

    train_window_start = time_periods_index[time_periods == train_start_period][0]
    train_window_end = time_periods_index[time_periods == prediction_period][0]

    rf_model_data = model_data.iloc[range(train_window_start, train_window_end), :]

    rf_model_data.fillna(0, inplace=True)
    d_train = lgb.Dataset(rf_model_data.loc[rf_model_data['Train'] == 1, 'X1_avg':'X70_avg_pctile'],
                          label=rf_model_data.loc[rf_model_data['Train'] == 1, 'Norm_Ret_F6M'])
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              'application': 'regression',
              'metric': 'l2_root',
              'learning_rate': 0.1,
              'max_depth': 20,
              'num_leaves': 500}
    model = lgb.train(params, d_train, num_boost_round=20, valid_sets=[d_train])

    return (model)


def svm(train_start_period, prediction_period):
    train_window_start = time_periods_index[time_periods == train_start_period][0]
    train_window_end = time_periods_index[time_periods == prediction_period][0]

    rf_model_data = model_data.iloc[range(train_window_start, train_window_end), :]

    rf_model_data.fillna(0, inplace=True)

    Svc_classifier =SVC().fit(rf_model_data.loc[rf_model_data['Train'] == 1, 'X1_avg':'X70_avg_pctile'],
           rf_model_data.loc[rf_model_data['Train'] == 1, 'Norm_Ret_F6M'])


    return Svc_classifier



def calc_metrics(time_period, predicted_rank):
    # subset actual values for prediction time_period
    actuals = model_data.loc[(model_data['time_period'] == time_period) & (model_data['Train'] == 1), :]

    # join predictions onto actuals
    actuals['Rank_F6M_pred'] = predicted_rank

    # calculate spearman correlation
    spearman = stat.spearmanr(actuals['Rank_F6M'], actuals['Rank_F6M_pred'])[0]

    # calculate NDCG = DCG of Top 20% / Ideal DCG of Top 20%
    # subset top 20% predictions
    t20 = actuals.loc[actuals['Rank_F6M_pred'] <= np.nanpercentile(actuals['Rank_F6M_pred'], 20), :]
    t20['discount'] = np.amax(actuals['Rank_F6M_pred']) / (np.amax(actuals['Rank_F6M_pred']) + actuals['Rank_F6M_pred'])
    t20['gain'] = t20['Norm_Ret_F6M'] * t20['discount']
    DCG = np.sum(t20['gain'])

    # subset top 20% actuals
    i20 = actuals.loc[actuals['Rank_F6M'] <= np.nanpercentile(actuals['Rank_F6M'], 20), :]
    i20['discount'] = np.amax(actuals['Rank_F6M']) / (np.amax(actuals['Rank_F6M']) + actuals['Rank_F6M'])
    i20['gain'] = i20['Norm_Ret_F6M'] * i20['discount']
    IDCG = np.sum(i20['gain'])

    NDCG = DCG / IDCG

    # return time_period, spearman correlation, NDCG
    return (pd.DataFrame([(time_period, spearman, NDCG)], columns=['time_period','spearman','NDCG']))

#time = '2002_1'


model_data.fillna(0, inplace=True)
#train_results = pd.DataFrame(columns=['time_period', 'spearman', 'NDCG'])
def randomForest_evaluate():
 train_results = pd.DataFrame(columns=['time_period', 'spearman', 'NDCG'])

 for time in time_periods[11:]:
    rf = randomForest(train_start_period='1996_2', prediction_period=time)

    if (time != '2017_1'):
        train_predictions = rf.predict(
            model_data.loc[(model_data['time_period'] == time) & (model_data['Train'] == 1), 'X1_avg':'X70_avg_pctile'])
        train_rank_predictions = len(train_predictions) - stat.rankdata(train_predictions, method='ordinal').astype(
            int) + 1
        print("Train predictions: ",train_rank_predictions)
        train_results = train_results.append(calc_metrics(time_period=time, predicted_rank=train_rank_predictions))

    test_predictions = rf.predict(
        model_data.loc[(model_data['time_period'] == time) & (model_data['Train'] == 0), 'X1_avg':'X70_avg_pctile'])
    test_rank_predictions = len(test_predictions) - stat.rankdata(test_predictions, method='ordinal').astype(int) + 1
    print("Test predictions: ",test_rank_predictions)
    pred_template.loc[pred_template['time_period'] == time, 'Rank_F6M'] = test_rank_predictions

    print("Time period " + time + " completed.")

 print(train_results)
 print(pred_template)
 train_results.to_csv('train_results_rf.csv')
 pred_template.to_csv('final_pred_rf.csv')

#randomForest_evaluate()
def sequentual_evaluate():
    train_results = pd.DataFrame(columns=['time_period', 'spearman', 'NDCG'])
    for time in time_periods[11:]:
        rf = sequentalModel(train_start_period='1996_2', prediction_period=time)
        if (time != '2017_1'):
            train_predictions = rf.predict(
                model_data.loc[(model_data['time_period'] == time) & (model_data['Train'] == 1),
                'X1_avg':'X70_avg_pctile'])
            train_rank_predictions = len(train_predictions) - stat.rankdata(train_predictions, method='ordinal').astype(
                int) + 1

            train_results = train_results.append(calc_metrics(time_period=time, predicted_rank=train_rank_predictions))

        test_predictions = rf.predict(
            model_data.loc[(model_data['time_period'] == time) & (model_data['Train'] == 0), 'X1_avg':'X70_avg_pctile'])
        test_rank_predictions = len(test_predictions) - stat.rankdata(test_predictions, method='ordinal').astype(
            int) + 1
        pred_template.loc[pred_template['time_period'] == time, 'Rank_F6M'] = test_rank_predictions
    print(train_results)
    print(pred_template)
    train_results.to_csv('train_results_seq.csv')
    #pred_template.to_csv('final_pred_seq.csv')
def xgb_evaluate():
    train_results = pd.DataFrame(columns=['time_period', 'spearman', 'NDCG'])
    for time in time_periods[11:]:
        rf = xgBoost(train_start_period='1996_2', prediction_period=time)
        if (time != '2017_1'):
            train_predictions = rf.predict(
                xgb.DMatrix(model_data.loc[(model_data['time_period'] == time) & (model_data['Train'] == 1), 'X1_avg':'X70_avg_pctile']))
            train_rank_predictions = len(train_predictions) - stat.rankdata(train_predictions, method='ordinal').astype(
                int) + 1

            train_results = train_results.append(calc_metrics(time_period=time, predicted_rank=train_rank_predictions))

        test_predictions = rf.predict(
            xgb.DMatrix(model_data.loc[(model_data['time_period'] == time) & (model_data['Train'] == 0), 'X1_avg':'X70_avg_pctile']))
        test_rank_predictions = len(test_predictions) - stat.rankdata(test_predictions, method='ordinal').astype(
            int) + 1
        pred_template.loc[pred_template['time_period'] == time, 'Rank_F6M'] = test_rank_predictions

    print(train_results)
    print(pred_template)
    train_results.to_csv('train_results_xgb.csv')
    pred_template.to_csv('final_pred_xgb.csv')


def light_gmb_evaluate():
    train_results = pd.DataFrame(columns=['time_period', 'spearman', 'NDCG'])
    for time in time_periods[11:]:
        rf = light_gbm(train_start_period='1996_2', prediction_period=time)
        if (time != '2017_1'):
            train_predictions = rf.predict(
                model_data.loc[(model_data['time_period'] == time) & (model_data['Train'] == 1), 'X1_avg':'X70_avg_pctile'])
            train_rank_predictions = len(train_predictions) - stat.rankdata(train_predictions, method='ordinal').astype(
                int) + 1

            train_results = train_results.append(calc_metrics(time_period=time, predicted_rank=train_rank_predictions))

        test_predictions = rf.predict(
            model_data.loc[(model_data['time_period'] == time) & (model_data['Train'] == 0), 'X1_avg':'X70_avg_pctile'])
        test_rank_predictions = len(test_predictions) - stat.rankdata(test_predictions, method='ordinal').astype(
            int) + 1
        pred_template.loc[pred_template['time_period'] == time, 'Rank_F6M'] = test_rank_predictions
    print(train_results)
    print(pred_template)
    train_results.to_csv('train_results_gbm.csv')
    #pred_template.to_csv('final_pred_gbm.csv')


def svm_evaluate():
    train_results = pd.DataFrame(columns=['time_period', 'spearman', 'NDCG'])
    for time in time_periods[11:]:
        rf = svm(train_start_period='1996_2', prediction_period=time)
        if (time != '2017_1'):
            train_predictions = rf.predict(
                model_data.loc[(model_data['time_period'] == time) & (model_data['Train'] == 1),
                            'X1_avg':'X70_avg_pctile'])
            train_rank_predictions = len(train_predictions) - stat.rankdata(train_predictions, method='ordinal').astype(
                int) + 1

            train_results = train_results.append(calc_metrics(time_period=time, predicted_rank=train_rank_predictions))

        test_predictions = rf.predict(
            model_data.loc[(model_data['time_period'] == time) & (model_data['Train'] == 0),
                        'X1_avg':'X70_avg_pctile'])
        test_rank_predictions = len(test_predictions) - stat.rankdata(test_predictions, method='ordinal').astype(
            int) + 1
        pred_template.loc[pred_template['time_period'] == time, 'Rank_F6M'] = test_rank_predictions

    print(train_results)
    print(pred_template)
    train_results.to_csv('train_results_svm.csv')
    pred_template.to_csv('final_pred_svm.csv')


#print("RANDOM FOREST")
#randomForest_evaluate()
#print("LIGHT GBM")
#light_gmb_evaluate()
#print("SEQUENTUAL")
#sequentual_evaluate()
#print("XGB")
#xgb_evaluate()
#print("SVM")
#svm_evaluate()
def rankStocks(year):
    """years = ['2002_1', '2002_2', '2003_1', '2003_2', '2004_1', '2004_2', '2005_1', '2005_2', '2006_1', '2006_2',
             '2007_1', '2007_2', '2008_1', '2008_2', '2009_1',
             '2009_2', '2010_1', '2010_2', '2011_1', '2011_2', '2012_1', '2012_2', '2013_1', '2013_2', '2014_1',
             '2014_2', '2015_1',
             '2015_2', '2016_1', '2016_2','2017_1']"""
    df_rf = pd.read_csv("final_pred_rf.csv")
    df_gbm = pd.read_csv("final_pred_gbm.csv")
    df_seq = pd.read_csv("final_pred_seq.csv")
    df_xgb = pd.read_csv("final_pred_xgb.csv")
    df_svm = pd.read_csv("final_pred_svm.csv")
    #for year in years:
    df_rf1 = df_rf[df_rf['time_period'] == year]
    df_rf1=df_rf1.sort_values('Rank_F6M')
    df_gbm1 = df_gbm[df_gbm['time_period']==year]
    df_gbm1 = df_gbm1.sort_values('Rank_F6M')
    df_seq1 = df_seq[df_seq['time_period'] == year]
    df_seq1 = df_seq1.sort_values('Rank_F6M')
    df_xgb1 = df_xgb[df_xgb['time_period'] == year]
    df_xgb1 = df_xgb1.sort_values('Rank_F6M')
    df_svm1 = df_svm[df_svm['time_period'] == year]
    df_svm1 = df_svm1.sort_values('Rank_F6M')
    head = df_rf1.head(n=10)
    tail = df_rf1.tail(n=10)
    print(head)
    plt.figure(1)

    plt.scatter(df_rf1['Rank_F6M'], df_rf1['Rank_F6M'], color='blue')
    plt.scatter(head['Rank_F6M'],head['Rank_F6M'],color="green",label=head['index'])

    plt.scatter(tail['Rank_F6M'],tail['Rank_F6M'],color="red",label=tail['index'])

        #plt.xticks(df_rf1['index'])
    s="Random Forest prediction for period "+year
    plt.title(s)

    plt.legend()
    head = df_gbm1.head(n=10)
    tail = df_gbm1.tail(n=10)
    print(head)
    plt.figure(2)
    plt.scatter(df_gbm1['Rank_F6M'], df_gbm1['Rank_F6M'], color='blue')
    plt.scatter(head['Rank_F6M'], head['Rank_F6M'], color="green", label=head['index'])
    plt.scatter(tail['Rank_F6M'], tail['Rank_F6M'], color="red", label=tail['index'])

        # plt.xticks(df_rf1['index'])
    s = "LightGBM prediction for period " + year
    plt.title(s)

    plt.legend()
    plt.figure(3)
    head = df_seq1.head(n=10)
    tail = df_seq1.tail(n=10)
    print(head)
    plt.scatter(df_seq1['Rank_F6M'], df_seq1['Rank_F6M'], color='blue')
    plt.scatter(head['Rank_F6M'], head['Rank_F6M'], color="green", label=head['index'])
    plt.scatter(tail['Rank_F6M'], tail['Rank_F6M'], color="red", label=tail['index'])

        # plt.xticks(df_rf1['index'])
    s = "Deep Learning prediction for period " + year
    plt.title(s)

    plt.legend()
    plt.figure(4)
    head = df_xgb1.head(n=10)
    tail = df_xgb1.tail(n=10)
    print(head)
    plt.scatter(df_xgb1['Rank_F6M'], df_xgb1['Rank_F6M'], color='blue')
    plt.scatter(head['Rank_F6M'], head['Rank_F6M'], color="green", label=head['index'])

    plt.scatter(tail['Rank_F6M'], tail['Rank_F6M'], color="red", label=tail['index'])

        # plt.xticks(df_rf1['index'])
    s = "XGBoost prediction for period " + year
    plt.title(s)

    plt.legend()
    plt.figure(5)

    head = df_svm1.head(n=10)
    tail = df_svm1.tail(n=10)
    print(head)
    plt.scatter(df_svm1['Rank_F6M'], df_svm1['Rank_F6M'], color='blue')
    plt.scatter(head['Rank_F6M'], head['Rank_F6M'], color="green", label=head['index'])
    plt.scatter(tail['Rank_F6M'], tail['Rank_F6M'], color="red", label=tail['index'])
    # plt.xticks(df_rf1['index'])
    s = "SVM prediction for period " + year
    plt.title(s)

    plt.legend()

    plt.show()



#rankStocks('2017_2')
def spearnManByPeriod(year):
    tr_rf = pd.read_csv("train_results_rf.csv")
    tr_gbm = pd.read_csv("train_results_gbm.csv")
    tr_seq = pd.read_csv("train_results_seq.csv")
    tr_xgb = pd.read_csv("train_results_xgb.csv")
    tr_svm = pd.read_csv("train_results_svm.csv")
    spearman=[]
    tr_rf=tr_rf[tr_rf['time_period']==year]
    tr_gbm=tr_gbm[tr_gbm['time_period']==year]
    tr_seq = tr_seq[tr_seq['time_period'] == year]
    tr_xgb = tr_xgb[tr_xgb['time_period'] == year]
    tr_svm = tr_svm[tr_svm['time_period'] == year]
    #tr_gbm = tr_gbm[tr_gbm['time_period'] == year]
    spearman.append(tr_rf['spearman'].values)
    spearman.append(tr_gbm['spearman'].values)
    spearman.append(tr_seq['spearman'].values)
    spearman.append(tr_xgb['spearman'].values)
    spearman.append(tr_svm['spearman'].values)
    return spearman
def visualizeSpearmanByPeriod():
    """years = ['2002_1', '2002_2', '2003_1', '2003_2', '2004_1', '2004_2', '2005_1', '2005_2', '2006_1', '2006_2',
             '2007_1', '2007_2', '2008_1', '2008_2', '2009_1',
             '2009_2', '2010_1', '2010_2', '2011_1', '2011_2', '2012_1', '2012_2', '2013_1', '2013_2', '2014_1',
             '2014_2', '2015_1',
             '2015_2', '2016_1', '2016_2']"""
    years = ['2015_2', '2016_1', '2016_2']
    for i,year in enumerate(years):
        sp=spearnManByPeriod(year)
        print(sp)
        x=131+i
        plt.subplot(x)
        plt.plot(sp)
        plt.title(year)
        plt.annotate("RF", (0.0,sp[0]))
        plt.annotate("LGBM", (1.0, sp[1]))
        plt.annotate("DL", (2.0, sp[2]))
        plt.annotate("XGB", (3.0, sp[3]))
        plt.annotate("SVM", (4.0, sp[4]))
    plt.legend()

    plt.show()
visualizeSpearmanByPeriod()
def NDCGbyPeriod(year):
    tr_rf = pd.read_csv("train_results_rf.csv")
    tr_gbm = pd.read_csv("train_results_gbm.csv")
    tr_seq = pd.read_csv("train_results_seq.csv")
    tr_xgb = pd.read_csv("train_results_xgb.csv")
    tr_svm = pd.read_csv("train_results_svm.csv")
    spearman=[]
    tr_rf=tr_rf[tr_rf['time_period']==year]
    tr_gbm=tr_gbm[tr_gbm['time_period']==year]
    tr_seq = tr_seq[tr_seq['time_period'] == year]
    tr_xgb = tr_xgb[tr_xgb['time_period'] == year]
    tr_svm = tr_svm[tr_svm['time_period'] == year]
    #tr_gbm = tr_gbm[tr_gbm['time_period'] == year]
    spearman.append(tr_rf['NDCG'].values)
    spearman.append(tr_gbm['NDCG'].values)
    spearman.append(tr_seq['NDCG'].values)
    spearman.append(tr_xgb['NDCG'].values)
    spearman.append(tr_svm['NDCG'].values)
    return spearman
def visualizeNDCGByPeriod():
    years = ['2004_2']
    for year in years:
        sp=NDCGbyPeriod(year)
        print(sp)
        plt.plot(sp,label=year)
        plt.annotate("Random forest model", (0.0, sp[0]))
        plt.annotate("LightGBM model", (1.0, sp[1]))
        plt.annotate("Deep learning model", (2.0, sp[2]))
        plt.annotate("XGBoost model", (3.0, sp[3]))
        plt.annotate("SVM model", (4.0, sp[4]))
    plt.legend()
    plt.title("NDCG")
    plt.show()
#visualizeNDCGByPeriod()
def calc2002to2016():
 """years = ['2002_1','2002_2','2003_1','2003_2','2004_1','2004_2','2005_1','2005_2','2006_1','2006_2','2007_1','2007_2','2008_1','2008_2','2009_1',
         '2009_2','2010_1','2010_2','2011_1','2011_2','2012_1','2012_2','2013_1','2013_2','2014_1','2014_2','2015_1',
         '2015_2','2016_1','2016_2']"""
 years = ['2004_2', '2010_1']
 df_rf = pd.read_csv("final_pred_rf.csv")
 df_gbm = pd.read_csv("final_pred_gbm.csv")
 df_seq = pd.read_csv("final_pred_seq.csv")
 df_xgb = pd.read_csv("final_pred_xgb.csv")
 df_svm = pd.read_csv("final_pred_svm.csv")

 for year in years:
    index = data[data['time_period'] == year]
    index.fillna(0, inplace=True)
    idx = index[index['Train']==1]
    df_rf1 = df_rf[df_rf['time_period']==year]
    vrf = df_rf1['Rank_F6M'].values
    l=idx.shape[0]
    r=idx.shape[0]+vrf.shape[0]
    x = [i for i in range(l)]
    y = [i for i in range(l,r)]
    plt.plot(x[:],idx['Rank_F6M'],label="Train Data")
    plt.plot(y[:],vrf,label='Random Forest predictions')
    s="Predictions for period "+year
    plt.title(s)
    plt.legend()
    plt.show()
    index = data[data['time_period'] == year]
    index.fillna(0, inplace=True)
    idx = index[index['Train'] == 1]
    df_gbm1 = df_gbm[df_rf['time_period'] == year]
    vgbm = df_gbm1['Rank_F6M'].values
    l = idx.shape[0]
    r = idx.shape[0] + vgbm.shape[0]
    x = [i for i in range(l)]
    y = [i for i in range(l, r)]
    plt.plot(x[:], idx['Rank_F6M'], label="Train Data")
    plt.plot(y[:], vgbm, label='LightGBM predictions')
    s = "Predictions for period " + year
    plt.title(s)
    plt.legend()
    plt.show()
    index = data[data['time_period'] == year]
    index.fillna(0, inplace=True)
    idx = index[index['Train'] == 1]
    df_seq1 = df_seq[df_rf['time_period'] == year]
    vseq = df_seq1['Rank_F6M'].values
    l = idx.shape[0]
    r = idx.shape[0] + vseq.shape[0]
    x = [i for i in range(l)]
    y = [i for i in range(l, r)]
    plt.plot(x[:], idx['Rank_F6M'], label="Train Data")
    plt.plot(y[:], vseq, label='Deep learning predictions')
    s = "Predictions for period " + year
    plt.title(s)
    plt.legend()
    plt.show()
    index = data[data['time_period'] == year]
    index.fillna(0, inplace=True)
    idx = index[index['Train'] == 1]
    df_xgb1 = df_xgb[df_xgb['time_period'] == year]
    vxgb = df_xgb1['Rank_F6M'].values
    l = idx.shape[0]
    r = idx.shape[0] + vxgb.shape[0]
    x = [i for i in range(l)]
    y = [i for i in range(l, r)]
    plt.plot(x[:], idx['Rank_F6M'], label="Train Data")
    plt.plot(y[:], vxgb, label='XGBoost predictions')
    s = "Predictions for period " + year
    plt.title(s)
    plt.legend()
    plt.show()
    index = data[data['time_period'] == year]
    index.fillna(0, inplace=True)
    idx = index[index['Train'] == 1]
    df_svm1 = df_svm[df_svm['time_period'] == year]
    vsvm = df_svm1['Rank_F6M'].values
    l = idx.shape[0]
    r = idx.shape[0] + vsvm.shape[0]
    x = [i for i in range(l)]
    y = [i for i in range(l, r)]
    plt.plot(x[:], idx['Rank_F6M'], label="Train Data")
    plt.plot(y[:], vsvm, label='SVM predictions')
    s = "Predictions for period " + year
    plt.title(s)
    plt.legend()
    plt.show()
    y1 = [i for i in range(vseq.shape[0])]
    plt.plot(y1, vrf, label="Random forest prediction")
    plt.plot(y1, vgbm, label="LightGBM prediction")
    plt.plot(y1,vseq,label="Deep learning prediction")
    plt.plot(y1,vxgb,label="XGBoost prediction")
    plt.plot(y1, vsvm, label="SVM prediction")
    s1 = "Comparison of models for period "+year
    plt.title(s1)
    plt.legend()
    plt.show()


def calculate2017(year='2017_1'):
    df_rf = pd.read_csv("final_pred_rf.csv")
    df_gbm = pd.read_csv("final_pred_gbm.csv")
    df_seq = pd.read_csv("final_pred_seq.csv")
    df_xgb = pd.read_csv("final_pred_xgb.csv")
    df_svm = pd.read_csv("final_pred_svm.csv")
    x = pred_template[pred_template['time_period']=='2017_1']
    df_rf1 = df_rf[df_rf['time_period']==year]
    vrf = df_rf1['Rank_F6M'].values

    r=vrf.shape[0]

    y = [i for i in range(r)]

    plt.plot(y[:],vrf,label='Random Forest predictions')
    plt.plot(y,x['Rank_F6M'],label='Actual values')
    s="Predictions for period "+year
    plt.title(s)
    plt.legend()
    plt.show()

    df_gbm1 = df_gbm[df_rf['time_period'] == year]
    vgbm = df_gbm1['Rank_F6M'].values

    r = vgbm.shape[0]

    y = [i for i in range(r)]

    plt.plot(y[:], vgbm, label='LightGBM predictions')
    plt.plot(y,x['Rank_F6M'],label='Actual values')

    s = "Predictions for period " + year
    plt.title(s)
    plt.legend()
    plt.show()

    df_seq1 = df_seq[df_rf['time_period'] == year]
    vseq = df_seq1['Rank_F6M'].values

    r =vseq.shape[0]

    y = [i for i in range(r)]

    plt.plot(y[:], vseq, label='Deep learning predictions')
    plt.plot(y,x['Rank_F6M'],label='Actual values')

    s = "Predictions for period " + year
    plt.title(s)
    plt.legend()
    plt.show()

    df_xgb1 = df_xgb[df_xgb['time_period'] == year]
    vxgb = df_xgb1['Rank_F6M'].values

    r = vxgb.shape[0]

    y = [i for i in range(0, r)]

    plt.plot(y[:], vxgb, label='XGBoost predictions')
    plt.plot(y,x['Rank_F6M'],label='Actual values')

    s = "Predictions for period " + year
    plt.title(s)
    plt.legend()
    plt.show()

    df_svm1 = df_svm[df_svm['time_period'] == year]
    vsvm = df_svm1['Rank_F6M'].values

    r =vsvm.shape[0]

    y = [i for i in range(r)]

    plt.plot(y[:], vsvm, label='SVM predictions')
    plt.plot(y,x['Rank_F6M'],label='Actual values')

    s = "Predictions for period " + year
    plt.title(s)
    plt.legend()
    plt.show()
    y1 = [i for i in range(vseq.shape[0])]
    plt.plot(y1, vrf, label="Random forest prediction")
    plt.plot(y1, vgbm, label="LightGBM prediction")
    plt.plot(y1,vseq,label="Deep learning prediction")
    plt.plot(y1,vxgb,label="XGBoost prediction")
    plt.plot(y1, vsvm, label="SVM prediction")
    s1 = "Comparison of models for period "+year
    plt.title(s1)
    plt.legend()
    plt.show()
#calculate2017()
#allSpearman=[]
#allNDCGValues=[]
def spearmanAndNDCG():
    years = ['2002_1', '2002_2', '2003_1', '2003_2', '2004_1', '2004_2', '2005_1', '2005_2', '2006_1', '2006_2',
             '2007_1', '2007_2', '2008_1', '2008_2', '2009_1',
             '2009_2', '2010_1', '2010_2', '2011_1', '2011_2', '2012_1', '2012_2', '2013_1', '2013_2', '2014_1',
             '2014_2', '2015_1',
             '2015_2', '2016_1', '2016_2']
    tr_rf = pd.read_csv("train_results_rf.csv")
    tr_gbm = pd.read_csv("train_results_gbm.csv")
    tr_seq = pd.read_csv("train_results_seq.csv")
    tr_xgb = pd.read_csv("train_results_xgb.csv")
    tr_svm = pd.read_csv("train_results_svm.csv")
    spearmanValues=[]
    NDCGValues = []
    allSpearman=[]
    allNDCGValues=[]
    for year in years:
        tr_rf1 = tr_rf[tr_rf['time_period']==year]
        x=tr_rf1['spearman'].values
        y=year
        z=(x,y)
        spearmanValues.append(z)
        x=tr_rf1['NDCG']
        z=(x,y)
        NDCGValues.append(z)

    plt.figure(1,figsize=(10,10))
    for x,y in spearmanValues:
        plt.subplot(121)
        plt.scatter(x,x,label=y)
    plt.title("Spearman values using Random Forest")
    plt.legend()
    for x,y in NDCGValues:
        plt.subplot(122)
        plt.scatter(x,x,label=y)
    plt.title("NDCG values using Random Forest")
    plt.legend()
    plt.show()
    plt.figure(2,figsize=(10,10))
    plt.subplot(121)
    plt.plot(spearmanValues[0])
    plt.title("Spearnman function using Random Forest")
    plt.subplot(122)
    plt.plot(NDCGValues[0])
    plt.title("NDCG function using Random Forest")
    plt.show()
    allSpearman.append(spearmanValues)
    allNDCGValues.append(NDCGValues)
    for year in years:
        tr_gbm1 = tr_gbm[tr_gbm['time_period']==year]
        x=tr_gbm1['spearman'].values
        y=year
        z=(x,y)
        spearmanValues.append(z)
        x=tr_gbm1['NDCG']
        z=(x,y)
        NDCGValues.append(z)

    plt.figure(1,figsize=(10,10))
    for x,y in spearmanValues:
        plt.subplot(121)
        plt.scatter(x,x,label=y)
    plt.title("Spearman values using LightGBM")
    plt.legend()
    for x,y in NDCGValues:
        plt.subplot(122)
        plt.scatter(x,x,label=y)
    plt.title("NDCG values using LightGBM")
    plt.legend()
    plt.show()
    plt.figure(2,figsize=(10,10))
    plt.subplot(121)
    plt.plot(spearmanValues[0])
    plt.title("Spearnman function using LightGBM")
    plt.subplot(122)
    plt.plot(NDCGValues[0])
    plt.title("NDCG function using LightGBM")
    plt.show()
    allSpearman.append(spearmanValues)
    allNDCGValues.append(NDCGValues)
    for year in years:
        tr_seq1 = tr_seq[tr_seq['time_period']==year]
        x=tr_seq1['spearman'].values
        y=year
        z=(x,y)
        spearmanValues.append(z)
        x=tr_rf1['NDCG']
        z=(x,y)
        NDCGValues.append(z)

    plt.figure(1,figsize=(10,10))
    for x,y in spearmanValues:
        plt.subplot(121)
        plt.scatter(x,x,label=y)
    plt.title("Spearman values using Deep learning")
    plt.legend()
    for x,y in NDCGValues:
        plt.subplot(122)
        plt.scatter(x,x,label=y)
    plt.title("NDCG values using Deep learning")
    plt.legend()
    plt.show()
    plt.figure(2,figsize=(10,10))
    plt.subplot(121)
    plt.plot(spearmanValues[0])
    plt.title("Spearnman function using Deep learning")
    plt.subplot(122)
    plt.plot(NDCGValues[0])
    plt.title("NDCG function using Deep learning")
    plt.show()
    allSpearman.append(spearmanValues)
    allNDCGValues.append(NDCGValues)
    for year in years:
        tr_xgb1 = tr_xgb[tr_xgb['time_period']==year]
        x=tr_xgb1['spearman'].values
        y=year
        z=(x,y)
        spearmanValues.append(z)
        x=tr_rf1['NDCG']
        z=(x,y)
        NDCGValues.append(z)

    plt.figure(1,figsize=(10,10))
    for x,y in spearmanValues:
        plt.subplot(121)
        plt.scatter(x,x,label=y)
    plt.title("Spearman values using XGBoost")
    plt.legend()
    for x,y in NDCGValues:
        plt.subplot(122)
        plt.scatter(x,x,label=y)
    plt.title("NDCG values using XGBoost")
    plt.legend()
    plt.show()
    plt.figure(2,figsize=(10,10))
    plt.subplot(121)
    plt.plot(spearmanValues[0])
    plt.title("Spearnman function using XGBoost")
    plt.subplot(122)
    plt.plot(NDCGValues[0])
    plt.title("NDCG function using XGBoost")
    plt.show()
    allSpearman.append(spearmanValues)
    allNDCGValues.append(NDCGValues)
    for year in years:
        tr_svm1 = tr_svm[tr_svm['time_period']==year]
        x=tr_svm1['spearman'].values
        y=year
        z=(x,y)
        spearmanValues.append(z)
        x=tr_rf1['NDCG']
        z=(x,y)
        NDCGValues.append(z)

    plt.figure(1,figsize=(10,10))
    for x,y in spearmanValues:
        plt.subplot(121)
        plt.scatter(x,x,label=y)
    plt.title("Spearman values using SVM")
    plt.legend()
    for x,y in NDCGValues:
        plt.subplot(122)
        plt.scatter(x,x,label=y)
    plt.title("NDCG values using SVM")
    plt.legend()
    plt.show()
    plt.figure(2,figsize=(10,10))
    plt.subplot(121)
    plt.plot(spearmanValues[0])
    plt.title("Spearnman function using SVM")
    plt.subplot(122)
    plt.plot(NDCGValues[0])
    plt.title("NDCG function using SVM")
    plt.show()
    allSpearman.append(spearmanValues)
    allNDCGValues.append(NDCGValues)
    print("ALLS: ",len(allSpearman))
    print("ALLN: ",len(allNDCGValues))
    for i in allSpearman:
        plt.plot(i[0])
    plt.title("All spearman functions")
    plt.show()
    for i in allNDCGValues:
        plt.plot(i[0])
    plt.title("All NDCG functions")
    plt.show()

#spearmanAndNDCG()
#calc2002to2016()

"""
df_rf = pd.read_csv("final_pred_rf.csv")
df_gbm = pd.read_csv("final_pred_gbm.csv")
df_seq = pd.read_csv("final_pred_seq.csv")
df_xgb = pd.read_csv("final_pred_xgb.csv")
df_svm = pd.read_csv("final_pred_svm.csv")
df_rf=df_rf[df_rf['time_period']=='2017_1']
df_gbm=df_gbm[df_gbm['time_period']=='2017_1']
df_seq=df_seq[df_seq['time_period']=='2017_1']
df_xgb=df_xgb[df_xgb['time_period']=='2017_1']
df_svm=df_svm[df_svm['time_period']=='2017_1']
ranks17=[]
ranks17.append(df_rf['Rank_F6M'])
ranks17.append(df_gbm['Rank_F6M'])
ranks17.append(df_seq['Rank_F6M'])
ranks17.append(df_xgb['Rank_F6M'])
ranks17.append(df_svm['Rank_F6M'])
for i,m in enumerate(ranks17):
    for j,n in enumerate(ranks17):
        plt.plot(m,label=i)
        plt.plot(n,label=j)
        print("Spearman between ",i," and ",j," is: ",stat.spearmanr(m,n))
        plt.legend()
        plt.show()
"""
"""
tr_rf = pd.read_csv("train_results_rf.csv")
tr_gbm = pd.read_csv("train_results_gbm.csv")
tr_seq = pd.read_csv("train_results_seq.csv")
tr_xgb = pd.read_csv("train_results_xgb.csv")
tr_svm = pd.read_csv("train_results_svm.csv")
plt.plot(tr_rf['spearman'],label="Spearman RF")
#plt.plot(tr_gbm['spearman'],label="Spearman GBM")
#plt.plot(tr_seq['spearman'],label="Spearman DL")
#plt.plot(tr_xgb['spearman'],label="Spearman XGB")
#plt.plot(tr_svm['spearman'],label='Spearman SVM')
#plt.xlabel([tr_rf['time_period']])
plt.legend()
plt.show()
"""