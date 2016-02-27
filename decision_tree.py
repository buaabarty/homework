import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate
import statsmodels.api as sm
import sklearn.feature_selection as feature_selection
import sys
import sklearn.tree as tree
import sklearn.cross_validation as cross_validation

if __name__ == "__main__":
    # read data
    result = pd.read_csv('bigdata.csv')

    print result.dtypes

    result['result'] = result['result'].astype('category')
    result['socialsecurity '] = result['socialsecurity'].astype('category')
    result['user_has_car '] = result['user_has_car'].astype('category')
    result['qid77 '] = result['qid77'].astype('category')
    result['quality '] = result['quality'].astype('category')
    result['id '] = result['id'].astype('category')
    result['married '] = result['married'].astype('category')
    result['tax '] = result['tax'].astype('category')
    result['house '] = result['house'].astype('category')

    # print feature plot
#    fig = plt.figure(1, figsize=(30, 30))
#    ax = fig.add_subplot(111)
#    bp = plt.boxplot(result['term'])
#    fig.savefig('fig1.png', bbox_inches='tight')

    # feature selection
    corrmatrix = result.corr(method='spearman')

    result['result'] = result['result'].astype('int')
    result['user_loan_experience'] = result['user_loan_experience'].astype('int')

    for key in result:
        try:
            print key, feature_selection.chi2(result[['result', key]], result['result'])
        except:
            continue

    # build model
    model_data = result[['result','term','limit','user_has_car','user_loan_experience','qid77','cash_receipts','user_income_by_card','user_work_period','qid123',\
                                'qid57','car_value','spam_score','mobile_verify','quality','id','married','house','tax']]
    print model_data.head()
    target = model_data['result']
    data = model_data.ix[:,'limit':]
    train_data, test_data, train_target, test_target = cross_validation.train_test_split(\
        data, target, test_size=0.4,train_size=0.6, random_state=12345)

    # decision tree
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=5)
    clf.fit(train_data,train_target)

    #view prediction model
    train_est = clf.predict(train_data)
    train_est_p = clf.predict_proba(train_data)[:,1]
    test_est = clf.predict(test_data)
    test_est_p = clf.predict_proba(test_data)[:,1]
    print pd.DataFrame({'test_target':test_target, 'test_est':test_est, 'test_est_p':test_est_p}).T
