import os
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import *
from imutils import paths
import face_recognition
import pickle
import numpy as np
import cv2
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

files_path = 'files'
model_path = f'{files_path}/FER_trained_model.pt'
cv2_xml_path = f'{files_path}/haarcascade_frontalface_default.xml'
model = Face_Emotion_CNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
model.eval()
emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                4: 'anger', 5: 'disguest', 6: 'fear'}

app = Flask(__name__)

@app.route('/detect/<device_id>', methods=['POST','GET'])
def upload(device_id):
    try:
        os.system('mkdir {}/{}'.format(files_path, device_id))
        file = request.get_json()['image']
        nparr = np.array(file, dtype=np.uint8)
        names, predictions = detect(device_id, nparr)
        print(names)
        print(predictions)
        if len(predictions) == 0:
            filename = '{}/{}/photo_{}.jpg'.format(files_path, device_id, time.time())
            print('Image shape is:', nparr.shape)
            cv2.imwrite(filename, nparr)
            return jsonify({"emotions": -1, "found": False})
        detection = {"emotions": predictions, "names": names, "found": True}
        filename = '{}/{}/photo_{}.jpg'.format(files_path, device_id, time.time())
        print('Image shape is:', nparr.shape)
        cv2.imwrite(filename, nparr)
        return jsonify(detection)
    except Exception as e:
        print(str(e))
        return jsonify({"emotions": -1, "found": False})

@app.route("/register/<device_id>/<user_id>", methods=['get', 'post'])
def register(device_id, user_id):
    try:
        file  = request.files['file']
        nparr = np.fromstring(file.read(), np.uint8)
        nparr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        device_file_path = '{}/{}'.format(files_path, device_id)
        os.system('mkdir {}'.format(device_file_path))
        os.system('mkdir {}/dataset'.format(device_file_path))
        os.system('mkdir {}/dataset/{}'.format(device_file_path, user_id))
        h = nparr.shape[0]
        w = nparr.shape[1]
        max_dim = max(h,w)
        if max_dim > 900:
            if max_dim == h:
                w = int(900 * w / h) 
                h = 900
            else:
                h = int(900 * h / w)
                w = 900
        img = cv2.resize(nparr, (w,h), interpolation = cv2.INTER_AREA)
        filename = '{}/dataset/{}/photo_{}.jpg'.format(device_file_path, user_id, time.time())
        cv2.imwrite(filename, img)
        response = {"status": 200}
        return jsonify(response)
    except Exception as e:
        print(str(e))
        return jsonify({"status": 400})

@app.route("/register_confirm/<device_id>", methods=['get', 'post'])
def register_confirm(device_id):
    try:
        encode_images(device_id)
        return jsonify({"status": 200})
    except Exception:
        return jsonify({"status": 400})

def detect(device_id, frame):
    boxes, names = recognize_image(device_id, frame)
    val_transform = transforms.Compose([
        transforms.ToTensor()])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []
    if len(boxes) == 0:
        return [], []
    for (top, right, bottom, left) in boxes:
        resize_frame = cv2.resize(gray[top:bottom, left:right], (48, 48))
        X = resize_frame/256
        X = Image.fromarray(X)
        X = val_transform(X).unsqueeze(0)
        with torch.no_grad():
            log_ps = model(X.float())
            ps = torch.exp(log_ps)
            print(ps)
            results.append(ps.numpy()[0].tolist())
    return names, results

def encode_images(device_id):
    print("[INFO] quantifying faces...")
    user_image_path = '{}/{}/dataset/'.format(files_path, device_id)
    encoding_path = '{}/{}/encoding.pickle'.format(files_path, device_id)
    imagePaths = list(paths.list_images(user_image_path))
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb,
            model="cnn")

        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(encoding_path, "wb")
    f.write(pickle.dumps(data))
    f.close()

def recognize_image(device_id, image):
    print("[INFO] loading encodings...")
    encoding_path = '{}/{}'.format(files_path, device_id)
    encoding_found = os.path.isfile("{}/encoding.pickle".format(encoding_path))
    if encoding_found:
        data = pickle.loads(open("{}/encoding.pickle".format(encoding_path), "rb").read())
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb,
        model="cnn")
     
    names = []
    if not encoding_found:
        for box in boxes:
            names.append("Unknown")
        print("No encoding")
        print(boxes)
        print(names)
        return boxes, names

    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        
        names.append(name)

    return boxes, names

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='3000', debug=True)

            