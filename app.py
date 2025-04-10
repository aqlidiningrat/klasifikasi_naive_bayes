from flask import Flask, render_template, url_for, request, redirect, send_file
from werkzeug.utils import secure_filename
import numpy as np
import cv2 as cv
import requests
import os, random

app = Flask(__name__)
root = os.getcwd()
path = os.path.join(root) # mysite
# load the haarcascade xml file
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# app.config
app.secret_key = 'naive_bayes klasifikasi jenis kelamin citra wajah'
app.config['UPLOAD_FOLDER'] = os.path.join(path,'static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #batas ukuran gambar 16mb
ALLOWED_EXTENSIONS = set(['jpeg','jpg','png'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

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
    hist, bins = np.histogram(hist, bins=10)
    print('hist_hog:', hist)
    print('bins_hog:', bins)
    return hist

def klasifikasi_naive_bayes(face):
    fiturHOG = ekstraksiCiriHOG(face)
    strUrlParams = '-'.join([str(i) for i in fiturHOG])
    responAPI = requests.get('https://wanDinulAqli.pythonanywhere.com/'+'dataset10bin_hog.xlsx/'+strUrlParams)
    print('fiturHOG:',fiturHOG, type(fiturHOG), fiturHOG.shape)
    print('strUrlParams:', strUrlParams, type(strUrlParams))
    print(responAPI.text, str(responAPI), '\n')
    return responAPI.text

def deteksiWajah(filename):
    img = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'inputImage', filename)) # read the image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # preprocessing image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) # detect faces within image
    if (len(faces) == 0):
        # save the outputImage
        nameOutputImage = '0wajah_'+filename
        cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],'outputImage/'+nameOutputImage), img)
        return len(faces), [len(faces)], 'cobaFileGambarLain!!', nameOutputImage

    m = 0
    for (x,y,w,h) in faces:
        m = m+1
        face = img[y:y+h, x:x+w]
        # save the faces
        nameFaces = 'face'+str(m)+'_'+filename
        cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],'faces/'+nameFaces), face)

    m, ind, ffc, hkl = 0, [], [], []
    for (x,y,w,h) in faces:
        m = m+1
        print('-------------------wajah',m,'dari',len(faces),'wajahTerdeteksi')
        face = img[y:y+h, x:x+w]
        # run klasifikasi_naive_bayes
        h_predict = klasifikasi_naive_bayes(face)

        # draw rectangles around the faces
        colorRectandText = (242,51,70) if h_predict=='Laki-Laki' else (132,22,254)
        cv.rectangle(img, (x,y),(x+w, y+h), colorRectandText, 2)
        cv.putText(img, h_predict, (x+5,y+10), cv.FONT_HERSHEY_DUPLEX, 0.3, colorRectandText, 1)
        cv.putText(img, 'wajah '+str(m), (x+5,y+h-5), cv.FONT_HERSHEY_DUPLEX, 0.3, colorRectandText, 1)
        # create seriesLabels
        ind.append(int(m-1)), ffc.append(str(m)), hkl.append(h_predict)

    # save the outputImage
    nameOutputImage = str(m)+'wajah_'+filename
    cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],'outputImage/'+nameOutputImage), img)
    # set return seriesLabels
    seriesLabels = np.array([ffc, hkl])
    return len(faces), ind, seriesLabels, nameOutputImage

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def home2():
    if 'file' not in request.files:
        return render_template('index.html') # no name file in form html

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html') # no file chosen

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # save the inputImages
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'inputImage', filename))
        jlhwajah, ind, seriesLabels, filenameOutputImage = deteksiWajah(filename)
        if (jlhwajah == 0):
            return render_template('index.html', lenFaces=str(jlhwajah), seriesLabels=seriesLabels, filenameOutputImage=filenameOutputImage)

        return render_template('index.html', lenFaces=jlhwajah, ind=ind, seriesLabels=seriesLabels, filenameOutputImage=filenameOutputImage, filename=filename)

    else:
        return render_template('index.html', lenFaces='yangDiupload bukanFileGambar', filename=file.filename)

@app.route('/displayOutputImage/<filename>')
def displayOutputImage(filename):
    return redirect(url_for('static', filename='outputImage/'+filename))

@app.route('/displayFaces/<filename>')
def displayFaces(filename):
    return redirect(url_for('static', filename='faces/'+filename))

@app.route('/downloadOutputImage/<filename>')
def downloadOutputImage(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'],'outputImage/'+filename), as_attachment=True)

@app.route('/downloadFaces/<filename>')
def downloadFaces(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'],'faces/'+filename), as_attachment=True)

if __name__=='__main__':
    app.run(debug=True)
