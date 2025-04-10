# from flask import Flask
# import sklearn.model_selection as ms
# import sklearn.preprocessing as pp
# import sklearn.metrics as met
# import pandas as pd
# import numpy as np
# import os
#
# app = Flask(__name__)
# root = os.getcwd()
# path = os.path.join(root) # mysite
#
# def r_default():
#     return 'app_api.pythonanywhere.com/namaDS/fitur0-fitur1-fitur2-danSeterusnya...'
#
# def klasifikasi_naive_bayes(namaDS, fitur):
#     df = pd.read_excel(os.path.join(path, namaDS))
#     X = df.drop(['fname','label'], axis=1)
#     y = df['label']
#
#     # train_test_split
#     X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)
#     # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#
#     # StandardScaler
#     scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
#     scl.fit(X_train)
#     X_train = scl.transform(X_train)
#     X_test = scl.transform(X_test)
#     # print(X_train.min(), X_train.max(), '||', X_test.min(), X_test.max())
#
#     # naive_bayes
#     import sklearn.naive_bayes as nb
#     model = nb.GaussianNB()
#     model.fit(X_train, y_train)
#     y_predict = model.predict(X_test)
#
#     # classification_report
#     accuracy = met.accuracy_score(y_test, y_predict)
#     confusionmatrix = met.confusion_matrix(y_test, y_predict)
#     precision = met.precision_score(y_test, y_predict)
#     sensitifity = met.recall_score(y_test, y_predict)
#     report = met.classification_report(y_test, y_predict)
#     # print(report)
#
#     h_predict = model.predict(fitur)
#     h = 'Laki-Laki' if h_predict == [0] else 'Perempuan'
#     return h
#
# def arahAPI(namaDS, fitur):
#     if (namaDS == 'dataset10bin_hog.xlsx'):
#         f = fitur.split('-')
#         fitur = np.array([[int(i) for i in f]])
#         if len(fitur[0]) != 10:
#             return r_default()
#             # endif
#         return klasifikasi_naive_bayes(namaDS, fitur)
#
#     else:
#         return r_default()
#
# @app.route('/')
# def home():
#     return r_default()
#
# @app.route('/<namaDS>/<fitur>')
# def responAPI(namaDS,fitur):
#     respon = arahAPI(namaDS, fitur)
#     return respon
#
# if __name__=='__main__':
#     app.run(debug=True)

# testResponsAPI
def responAPI():
    import requests, random, numpy
    urlParams = numpy.array([random.randint(111,999) for i in range(0,10)])
    strUrlParams = '-'.join([str(i) for i in urlParams])
    responAPI = requests.get('https://wanDinulAqli.pythonanywhere.com'+'/dataset10bin_hog.xlsx'+'/'+strUrlParams)
    print(urlParams, type(urlParams))
    print(strUrlParams)
    return responAPI.text

if __name__=='__main__':
    h = responAPI()
    print(h, type(h))
