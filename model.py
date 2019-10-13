
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm_notebook as tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter





#  CATBOOST Model
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from catboost import Pool, CatBoostClassifier
from category_encoders import TargetEncoder
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model', n_folds=5):
    kf = StratifiedKFold(n_splits=n_folds, shuffle = True, random_state = 228)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = test.columns
    i = 1
    for dev_index, val_index in fold_splits:
        print('-------------------------------------------')
        print('Started ' + label + ' fold ' + str(i) + f'/{n_folds}')
        dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
        dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, fi = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        feature_importances[f'fold_{i}'] = fi
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            print(label + ' cv score {}: {}'.format(i, cv_score), '\n')
        i += 1
    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / n_folds
    results = {'label': label,
              'train': pred_train, 'test': pred_full_test,
              'cv': cv_scores, 'fi': feature_importances}
    return results


def runCAT(train_X, train_y, test_X, test_y, test_X2, params):
    # Pool the data and specify the categorical feature indices
    print('Pool Data')
    _train = Pool(train_X, label=train_y)
    _valid = Pool(test_X, label=test_y)    
    print('Train CAT')
    model = CatBoostClassifier(**params)
    fit_model = model.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=1000,
                          plot=False)
    feature_im = fit_model.feature_importances_
    print('Predict 1/2')
    pred_test_y = fit_model.predict_proba(test_X)[:, 1]
    print('Predict 2/2')
    pred_test_y2 = fit_model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2, feature_im


# Use some baseline parameters
cat_params = {'loss_function': 'CrossEntropy', 
              'eval_metric': "AUC",
              'learning_rate': 0.01,
              'iterations': 10000,
              'random_seed': 42,
              'od_type': "Iter",
              'early_stopping_rounds': 150,
             }

n_folds = 10
results = run_cv_model(train[train_cols].fillna(0), test[train_cols].fillna(0), train['redemption_status'], runCAT, cat_params, auc, 'cat', n_folds=n_folds)
day = 2
sub = 2
name = f"day_{day}_sub_{sub}"
tmp = dict(zip(test.id.values, results['test']))
answer1 = pd.DataFrame()
answer1['id'] = test.id.values
answer1['redemption_status'] = answer1['id'].map(tmp)
answer1.to_csv(f'{name}.csv', index = None)



#xgBOOST
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm_notebook as tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def standart_split(data, n_splits):
    split_list = []
    for i in range(n_splits):
        kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 228)
        for train_index, test_index in kf.split(data.iloc[:ltr, :], data['redemption_status'][:ltr]):
            split_list += [(train_index, test_index)]
    return split_list

split_list = standart_split(data, 1)

def xgb_train(data, target, ltr, train_cols, split_list, param, n_e = 10000, cat_col = None, verb_num = None, imp=False):
    pred = pd.DataFrame()
    pred_val = np.zeros(ltr)
    score = []
    j = 0
    train_pred = pd.DataFrame()
    models = []
    for i , (train_index, test_index) in enumerate(split_list):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        param['seed'] = i
        tr = xgb.DMatrix(sc.fit_transform(np.array(data[train_cols])[train_index]), np.array(data[target])[train_index])
        te = xgb.DMatrix(sc.fit_transform(np.array(data[train_cols])[test_index]), np.array(data[target])[test_index])
        tt = xgb.DMatrix(sc.fit_transform(np.array(data[train_cols]))[ltr:, :])
        evallist = [(tr, 'train'), (te, 'test')]
        bst = xgb.train(param, tr, n_e, evallist,
                        early_stopping_rounds=150, verbose_eval = verb_num)
        
        pred[str(i)] =bst.predict(tt)
        pred_val[test_index] = bst.predict(te)
        score += [metrics.roc_auc_score(np.array(data[target])[test_index], pred_val[test_index])]
        models.append(bst)
        print(i, 'MEAN: ', np.mean(score), 'LAST: ', score[-1])
    train_pred[str(j)] = pred_val
    ans = pd.Series( pred.mean(axis = 1).tolist())
    ans.name = 'xgb'
    return pred, score, train_pred, bst
params = {'eta': 0.05,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99}
prediction, scores, oof, model = xgb_train(data, 'redemption_status', ltr, train_cols,
                       split_list, params,  verb_num  = 250)

tmp = prediction.copy()
for col in tmp.columns:
    tmp[col] = tmp[col].rank()
tmp = tmp.mean(axis = 1)
tmp  =tmp / tmp.max()
day = 2
sub = 6
name = f"day_{day}_sub_{sub}"
tmp = dict(zip(test.id.values, tmp))
answer1 = pd.DataFrame()
answer1['id'] = test.id.values
answer1['redemption_status'] = answer1['id'].map(tmp)
answer1.to_csv(f'{name}.csv', index = None)


#LightGBM
data[train_cols] = data[train_cols].fillna(data[train_cols].mean())
train = data[data['redemption_status'].notnull()]
test = data[data['redemption_status'].isnull()]
data = pd.concat([train, test], sort=False).reset_index(drop = True)
ltr = len(train)
def get_importances(clfs):
    importances = [clf.feature_importance('gain') for clf in clfs]
    importances = np.vstack(importances)
    mean_gain = np.mean(importances, axis=0)
    features = clfs[0].feature_name()
    data = pd.DataFrame({'gain':mean_gain, 'feature':features})
    plt.figure(figsize=(8, 30))
    sns.barplot(x='gain', y='feature', data=data.sort_values('gain', ascending=False))
    plt.tight_layout()
    return data
def standart_split(data, n_splits):
    split_list = []
    for i in range(n_splits):
        kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 228)
        for train_index, test_index in kf.split(data.iloc[:ltr, :], data['redemption_status'][:ltr]):
            split_list += [(train_index, test_index)]
    return split_list

split_list = standart_split(data, 1)

def lgb_train(data, target, ltr, train_cols, split_list, param, n_e = 10000, cat_col = None, verb_num = None, imp=True):
    pred = pd.DataFrame()
    pred_val = np.zeros(ltr)
    score = []
    j = 0
    train_pred = pd.DataFrame()
    models = []
    for i , (train_index, test_index) in enumerate(split_list):
        param['seed'] = i
        tr = lgb.Dataset(np.array(data[train_cols])[train_index], np.array(data[target])[train_index])
        te = lgb.Dataset(np.array(data[train_cols])[test_index], np.array(data[target])[test_index], reference=tr)
        tt = lgb.Dataset(np.array(data[train_cols])[ltr:, :])
        evallist = [(tr, 'train'), (te, 'test')]
        bst = lgb.train(param, tr, num_boost_round = n_e,valid_sets = [tr, te], feature_name=train_cols,
                        early_stopping_rounds=150, verbose_eval = verb_num)
        pred[str(i)] =bst.predict(np.array(data[train_cols])[ltr:])
        pred_val[test_index] = bst.predict(np.array(data[train_cols])[test_index])
        score += [metrics.roc_auc_score(np.array(data[target])[test_index], pred_val[test_index])]
        models.append(bst)
        print(i, 'MEAN: ', np.mean(score), 'LAST: ', score[-1])
    if imp:
        get_importances(models)
        plt.show()
    train_pred[str(j)] = pred_val
    ans = pd.Series( pred.mean(axis = 1).tolist())
    ans.name = 'lgb'
    return pred, score, train_pred, bst

param_lgb = { 'boosting_type': 'gbdt', 'objective': 'binary', 'metric':'auc',
             'bagging_freq':1, 'subsample':1, 'feature_fraction': 0.7,
              'num_leaves': 8, 'learning_rate': 0.05, 'lambda_l1':5,'max_bin':255}

prediction, scores, oof, model = lgb_train(data, 'redemption_status', ltr, train_cols,
                       split_list, param_lgb,  verb_num  = 250)

tmp = prediction.copy()
for col in tmp.columns:
    tmp[col] = tmp[col].rank()
tmp = tmp.mean(axis = 1)
tmp  =tmp / tmp.max()
day = 1
sub = 3
name = f"day_{day}_sub_{sub}"
tmp = dict(zip(test.id.values, tmp))
answer1 = pd.DataFrame()
answer1['id'] = test.id.values
answer1['redemption_status'] = answer1['id'].map(tmp)
answer1.to_csv(f'{name}.csv', index = None)