import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.naive_bayes as nb
import sklearn.metrics as met
import pandas as pd
import numpy as np
import cv2 as cv
import os

def ekstraksiCiriHOG(img):
    print('==> ekstraksiCiriHOG')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (128,128))
    print(img.shape)
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n*ang / (2*np.pi))

    bin_cells = []
    mag_cells = []
    cellx, celly = 8,8

    for i in range(0, int(img.shape[0] / celly)):
        for j in range(0, int(img.shape[1] / cellx)):
            bin_cells.append(bin[i*celly: i*celly+celly, j+cellx: j*cellx+cellx])
            mag_cells.append(mag[i*celly: i*celly+celly, j+cellx: j*cellx+cellx])

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    # transform to hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)

    hist /= np.linalg.norm(hist) + eps
    print(hist.shape, len(hist))
    hist, bins = np.histogram(hist, bins=12)
    print('hist_hog:', hist)
    print('bins_hog:', bins)
    return hist

def createDataTrain(data_folder):
    # rpb0, rpb1, = [],[] # 2bin
    # rpb0, rpb1, rpb2 = [],[],[] # 3bin
    # rpb0, rpb1, rpb2, rpb3 = [],[],[],[] # 4bin
    # rpb0, rpb1, rpb2, rpb3, rpb4 = [],[],[],[],[] # 5bin
    # rpb0, rpb1, rpb2, rpb3, rpb4, rpb5 = [],[],[],[],[],[] # 6bin
    # rpb0, rpb1, rpb2, rpb3, rpb4, rpb5, rpb6 = [],[],[],[],[],[],[] # 7bin
    # rpb0, rpb1, rpb2, rpb3, rpb4, rpb5, rpb6, rpb7 = [],[],[],[],[],[],[],[] # 8bin
    # rpb0, rpb1, rpb2, rpb3, rpb4, rpb5, rpb6, rpb7, rpb8 = [],[],[],[],[],[],[],[],[] # 9bin
    # rpb0, rpb1, rpb2, rpb3, rpb4, rpb5, rpb6, rpb7, rpb8, rpb9 = [],[],[],[],[],[],[],[],[],[] # 10bin
    # rpb0, rpb1, rpb2, rpb3, rpb4, rpb5, rpb6, rpb7, rpb8, rpb9, rpb10 = [],[],[],[],[],[],[],[],[],[],[] # 11bin
    rpb0, rpb1, rpb2, rpb3, rpb4, rpb5, rpb6, rpb7, rpb8, rpb9, rpb10, rpb11 = [],[],[],[],[],[],[],[],[],[],[],[] # 12bin
    fname, label_wajah = [],[]

    in_data_folder = os.listdir(data_folder)
    print('in_data_folder:', in_data_folder)
    for label_folder in in_data_folder:
        print('\n ==>label_folder:',label_folder)
        label = 0 if label_folder == 'man' else 1 # ternary-operator
        in_label_folder = data_folder+'/'+label_folder
        name_files = os.listdir(in_label_folder)
        i = 0
        for file in name_files:
            i += 1
            path_gambar = in_label_folder+'/'+file
            print('img_path:',path_gambar)
            img_bgr = cv.imread(path_gambar)
            print('img_shape:',img_bgr.shape, 'img_class:', label, 'n_for: '+str(i)+' to '+str(len(name_files)))

            # ekstraksiCiriHOG
            fitur = ekstraksiCiriHOG(img_bgr)

            # rpb0.append(fitur[0]), rpb1.append(fitur[1]), # 2bin
            # rpb0.append(fitur[0]), rpb1.append(fitur[1]), rpb2.append(fitur[2]), # 3bin
            # rpb0.append(fitur[0]), rpb1.append(fitur[1]), rpb2.append(fitur[2]), rpb3.append(fitur[3]), # 4bin
            # rpb0.append(fitur[0]), rpb1.append(fitur[1]), rpb2.append(fitur[2]), rpb3.append(fitur[3]), rpb4.append(fitur[4]), # 5bin
            # rpb0.append(fitur[0]), rpb1.append(fitur[1]), rpb2.append(fitur[2]), rpb3.append(fitur[3]), rpb4.append(fitur[4]), rpb5.append(fitur[5]), # 6bin
            # rpb0.append(fitur[0]), rpb1.append(fitur[1]), rpb2.append(fitur[2]), rpb3.append(fitur[3]), rpb4.append(fitur[4]), rpb5.append(fitur[5]), rpb6.append(fitur[6]), # 7bin
            # rpb0.append(fitur[0]), rpb1.append(fitur[1]), rpb2.append(fitur[2]), rpb3.append(fitur[3]), rpb4.append(fitur[4]), rpb5.append(fitur[5]), rpb6.append(fitur[6]), rpb7.append(fitur[7]), # 8bin
            # rpb0.append(fitur[0]), rpb1.append(fitur[1]), rpb2.append(fitur[2]), rpb3.append(fitur[3]), rpb4.append(fitur[4]), rpb5.append(fitur[5]), rpb6.append(fitur[6]), rpb7.append(fitur[7]), rpb8.append(fitur[8]), # 9bin
            # rpb0.append(fitur[0]), rpb1.append(fitur[1]), rpb2.append(fitur[2]), rpb3.append(fitur[3]), rpb4.append(fitur[4]), rpb5.append(fitur[5]), rpb6.append(fitur[6]), rpb7.append(fitur[7]), rpb8.append(fitur[8]), rpb9.append(fitur[9]), # 10bin
            # rpb0.append(fitur[0]), rpb1.append(fitur[1]), rpb2.append(fitur[2]), rpb3.append(fitur[3]), rpb4.append(fitur[4]), rpb5.append(fitur[5]), rpb6.append(fitur[6]), rpb7.append(fitur[7]), rpb8.append(fitur[8]), rpb9.append(fitur[9]), rpb10.append(fitur[10]), # 11bin
            rpb0.append(fitur[0]), rpb1.append(fitur[1]), rpb2.append(fitur[2]), rpb3.append(fitur[3]), rpb4.append(fitur[4]), rpb5.append(fitur[5]), rpb6.append(fitur[6]), rpb7.append(fitur[7]), rpb8.append(fitur[8]), rpb9.append(fitur[9]), rpb10.append(fitur[10]), rpb11.append(fitur[11]), # 12bin
            fname.append(file), label_wajah.append(label)

    # dataset8bin_hog
    data = pd.DataFrame({'fname':fname,
                            'rpb0':rpb0,'rpb1':rpb1, # 2bin
                            # 'rpb0':rpb0,'rpb1':rpb1,'rpb2':rpb2, # 3bin
                            # 'rpb0':rpb0,'rpb1':rpb1,'rpb2':rpb2,'rpb3':rpb3, # 4bin
                            # 'rpb0':rpb0,'rpb1':rpb1,'rpb2':rpb2,'rpb3':rpb3,'rpb4':rpb4, # 5bin
                            # 'rpb0':rpb0,'rpb1':rpb1,'rpb2':rpb2,'rpb3':rpb3,'rpb4':rpb4,'rpb5':rpb5, # 6bin
                            # 'rpb0':rpb0,'rpb1':rpb1,'rpb2':rpb2,'rpb3':rpb3,'rpb4':rpb4,'rpb5':rpb5,'rpb6':rpb6, # 7bin
                            # 'rpb0':rpb0,'rpb1':rpb1,'rpb2':rpb2,'rpb3':rpb3,'rpb4':rpb4,'rpb5':rpb5,'rpb6':rpb6,'rpb7':rpb7, # 8bin
                            # 'rpb0':rpb0,'rpb1':rpb1,'rpb2':rpb2,'rpb3':rpb3,'rpb4':rpb4,'rpb5':rpb5,'rpb6':rpb6,'rpb7':rpb7,'rpb8':rpb8, # 9bin
                            # 'rpb0':rpb0,'rpb1':rpb1,'rpb2':rpb2,'rpb3':rpb3,'rpb4':rpb4,'rpb5':rpb5,'rpb6':rpb6,'rpb7':rpb7,'rpb8':rpb8,'rpb9':rpb9, # 10bin
                            # 'rpb0':rpb0,'rpb1':rpb1,'rpb2':rpb2,'rpb3':rpb3,'rpb4':rpb4,'rpb5':rpb5,'rpb6':rpb6,'rpb7':rpb7,'rpb8':rpb8,'rpb9':rpb9,'rpb10':rpb10, # 11bin
                            'rpb0':rpb0,'rpb1':rpb1,'rpb2':rpb2,'rpb3':rpb3,'rpb4':rpb4,'rpb5':rpb5,'rpb6':rpb6,'rpb7':rpb7,'rpb8':rpb8,'rpb9':rpb9,'rpb10':rpb10,'rpb11':rpb11, # 12bin
                            'label':label_wajah})
    print(data)
    data.to_excel('dataFitur/citraGabungan/dataset12bin_hog.xlsx', sheet_name='dataset12bin_hog', index=False)

def akurasi_model(df):
    X = df.drop(['fname','label'], axis=1)
    y = df['label']
    
    # if (len(X.axes[1]) == 10):
    #     X = df.drop(['fname','label','rpb9'], axis=1)
    # print(X.corr())

    # train_test_split
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # StandardScaler
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)
    # print(X_train.min(), X_train.max(), '||', X_test.min(), X_test.max())
    # naive_bayes
    model = nb.GaussianNB()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    # classification_report
    accuracy = met.accuracy_score(y_test, y_predict)
    confusionmatrix = met.confusion_matrix(y_test, y_predict)
    precision = met.precision_score(y_test, y_predict)
    sensitifity = met.recall_score(y_test, y_predict)
    report = met.classification_report(y_test, y_predict)
    print('accuracy:',round(accuracy,2))
    print(confusionmatrix)
    print('precision:',round(precision,2))
    print('sensitifity:',round(sensitifity,2))
    # print(report)

def run_klasifikasi(path):
    fitur = ['_hog']
    for ft in fitur:
        for i in range(2,13):
            print('=> path:',path+str(i)+'bin'+ft)
            try:
                df = pd.read_excel(path+str(i)+'bin'+ft+'.xlsx')
            except Exception as e:
                continue
            akurasi_model(df)
            # endFor
        # endFor

if __name__=='__main__':
    # createDataTrain('d:/dataset/mrinalini-man-woman-detection/citraGabungan')
    run_klasifikasi('dataFitur/citraGabungan/dataset')
