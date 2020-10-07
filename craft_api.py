import os
from random import choice
from flask import Flask, jsonify, request, render_template
import time, requests
from PIL import Image
from io import BytesIO
from flask_ngrok import run_with_ngrok
# from flask import send_file
# import io
# import uuid
# import os
# import time

from predict_box import CraftDetection
import cv2
import numpy as np
import params as pr
model = CraftDetection()

desktop_agents = [
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']


def random_headers():
    return {'User-Agent': choice(desktop_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}


def download_image(image_url):
    header = random_headers()
    response = requests.get(image_url, headers=header, stream=True, verify=False, timeout=5)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    #image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image


def jsonify_str(output_list):
    with app.app_context():
        with app.test_request_context():
            result = jsonify(output_list)
    return result


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["CACHE_TYPE"] = "null"
run_with_ngrok(app)


def create_query_result(input_url, results, error=None):
    if error is not None:
        results = 'Error: ' + str(error)
    query_result = {
        'results': results
    }
    return query_result

@app.route("/query_box", methods=['GET', 'POST'])
def query_box():
    try:
        if request.method == "GET":
            img_url = request.args.get('url', default='', type=str)
            if 'http' in img_url:
              img = download_image(img_url)
            else:
              img = Image.open(img_url).convert('RGB')
        else:
            data = request.get_data()
            img = Image.open(BytesIO(data)).convert('RGB')

    except Exception as ex:
        print(ex)
        return jsonify_str(create_query_result("", "", ex))

    img = np.array(img)
    boxes, img_draw, total_time = model.text_detect(img)

    boxes_dict = []
    for box in boxes:
        for c in box:
            box_dict = {'top_left': {'x': c[0], 'y': c[1]},
                        'top_right': {'x'}}


    result = {'time': total_time, 'boxes': boxes, 'total_box': len(boxes)}
    return result

@app.route("/query_display", methods=['GET', 'POST'])
def query_display):
    try:
        if request.method == "GET":
            img_url = request.args.get('url', default='', type=str)
            if 'http' in img_url:
              img = download_image(img_url)
            else:
              img = Image.open(img_url).convert('RGB')
        else:
            data = request.get_data()
            img = Image.open(BytesIO(data)).convert('RGB')

    except Exception as ex:
        print(ex)
        return jsonify_str(create_query_result("", "", ex))

    img = np.array(img)
    boxes, img_draw, total_time = model.text_detect(img)

    pil_im = Image.fromarray(img_draw)
    img_result_path = os.path.join("static", "tmp.jpg")
    pil_im.save(img_result_path)
    return render_template("index.html", img_result=img_result_path)


if __name__ == "__main__":
    # app.run(pr.host, pr.port, threaded=True, debug=True)
    #app.run(debug=False, port=os.getenv('PORT', 5000))
    app.run()
