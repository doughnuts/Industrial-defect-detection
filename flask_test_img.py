from flask import Flask,render_template,request,send_file
from PIL import Image
import io
from u2net_test import model_test
from label_mask import label_mask
import cv2

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("client.html")

@app.route("/predict",methods=['POST'])
def predict():
    if request.method =='POST':
        file = request.files['file']
        img_bytes = file.read()
        #把字节数据转成图片
        img = Image.open(io.BytesIO(img_bytes))
        img.save("test_data/example.jpg")
        model_test()
        test_img_path = "test_data/example.jpg"
        test_mask_path ="test_result/example.jpg"
        res = label_mask(test_img_path,test_mask_path)
        cv2.imwrite("test_label_mask/example.png",res)
        return send_file("test_label_mask/example.png",mimetype="image/gif")
    else:
        return ""

if __name__== "__main__":
    app.run(host="0.0.0.0",port=5000)