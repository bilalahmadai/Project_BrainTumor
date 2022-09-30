from flask import Flask,render_template, request

import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img
import cv2 as cv
import numpy as np
import base64
app = Flask(__name__)

s_model = tf.keras.models.load_model('brain_tumor_v1.h5')




def Takeimg(i):
    img=cv.imread(i)
    resize_img = cv.resize(img,(64,64))
    input_img=np.expand_dims(resize_img, axis=0)
    result=s_model.predict(input_img)
    n=np.where(result==1.0)
    
    pred=np.argmax(result)
    if pred==0:
        tumor_pred="not Tumor Detected"
    else:
        tumor_pred="Tumor Detected"
    return tumor_pred


@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def predict():
   
    imgfile=request.files['imagefile']
    imgpath= "templates/images/" +imgfile.filename
    imgfile.save(imgpath)
    # img=load_img(imgpath)
    p=Takeimg(imgpath)


    return render_template('index.html',prediction = p)

# # Handle POST request
# @app.route('/', methods=['POST'])
# def canvas():
#     # Recieve base64 data from the user form
#     # canvasdata = request.form['predict']
#     encoded_data = request.form['Predict']

#     # Decode base64 image to python array
#     nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
#     img = cv.imdecode(nparr, cv.IMREAD_COLOR)
#     processed_img=preproces(img)
#     imgr=cv.resize(processed_img,(28,28))
#     img=np.expand_dims(imgr,axis=0)
#     # Convert 3 channel image (RGB) to 1 channel image (GRAY)
#     # gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     # # Resize to (28, 28)
#     # gray_image = cv.resize(gray_image, (28, 28), interpolation=cv.INTER_LINEAR)

#     # # Expand to numpy array dimenstion to (1, 28, 28)
#     # img = np.expand_dims(gray_image, axis=0)

    # try:
    #     prediction = np.argmax(s_model.predict(img))
    #     print(f"Prediction Result : {str(prediction)}")
    #     return render_template('index.html', response=str(prediction), canvasdata=canvasdata, success=True)
    # except Exception as e:
    #     return render_template('index.html', response=str(e), canvasdata=canvasdata)
if __name__ == '__main__':
    app.run(port=3000,debug=True)

    # https://medium.com/analytics-vidhya/how-to-deploy-digit-recognition-model-into-drawing-app-6e59f82a199c