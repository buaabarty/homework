import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
import copy
import math


if __name__ == "__main__":
    # load data
    df = pd.read_csv("bigdata.csv")
    print df.head(5)

    train_cols = df.columns[1:]
    x = df["result"]

    print train_cols

    print df.describe()

    cols_to_keep = [ 'result','term','limit','user_has_car','user_loan_experience','qid77','cash_receipts','user_income_by_card','user_work_period','qid123',
                    'qid57','car_value','spam_score','mobile_verify','quality','id','married','house','tax']
    data = df[cols_to_keep]
    print data.head()

    # const dimension
    data['intercept'] = 1.0

    # feature selection
    train_cols = data.columns[1:]

    logit = sm.Logit(data['result'].values, data[train_cols].values)
    result = logit.fit()

    print 'result', result

    # build prediction set
    combos = copy.deepcopy(data)
    predict_cols = combos.columns[1:]
    combos['intercept'] = 1.0

    # predict
    combos['predict'] = result.predict(combos[predict_cols])

    # evaluation
    total = 0
    hit = 0
    for value in combos.values:
        predict = value[-1]
        if predict > 0.25:
            total += 1
            if int(value[0]) == 1:
                hit += 1

    # output result
    print 'Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total)
    print "result summary"
    print result.summary()
    print "params result"
    print np.exp(result.params)

    # odds ratios and 95% CI
    params = result.params
    conf = result.conf_int()
    print '2.5%\t97.5%\tOR'
    for i in xrange(len(params)):
        print '%f\t%f\t%f' % (math.exp(conf[i][0]), math.exp(conf[i][1]), math.exp(params[i]))
