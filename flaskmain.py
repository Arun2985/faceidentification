import asyncio
import json
import time
import face_recognition
import imutils
import redis
import numpy as np
import requests
from flask import Flask, request


app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, db=0)
url = "https://172.17.98.55/api/v2/agent-monitoring/face-status/"
headers = {
    'Content-Type': 'application/json'
}
vale = ""
#f = open("Arun.txt", "a")


def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]
    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)
    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_locations, face_encodings


@app.route('/upload/', methods=['POST'])
def hello_world():
    try:
        global url
        data = request.get_json(force=True)
        #print(data)
        #print(data["userId"])
        #print(data['userId'])
        #print(data['pic2'])
        Img = data["pic1"]
        Img2 = data["pic2"]
        id = data["userId"]
        print(Img)
        logo1 = imutils.url_to_image(Img)
        logo2 = imutils.url_to_image(Img2)
        #print(logo2)
        #logo1 = imutils.resize(logo1, width=480)
        #print(logo2)

        E_id = r.get(id)

        print("e_id",E_id)
        if E_id ==None:
            print("e_id is not found")
            ocations, encodings = get_face_embeddings_from_image(logo1)
            my_face_encoding = encodings[0]
            r.set(id, my_face_encoding.tobytes())
            E_id = r.get(id)
        #except:
        #    ocations, encodings = get_face_embeddings_from_image(logo1)
        #    my_face_encoding = encodings[0]
        #    r.set(id, my_face_encoding.tobytes())
        #    E_id = r.get(id)
        #print(E_id)
        image = np.fromstring(E_id, dtype=np.float)
        decoded = image.reshape((128,))
        locations, encodings = get_face_embeddings_from_image(logo2)
        unknown_face_encoding = encodings[0]
        #dist = np.linalg.norm(decoded - unknown_face_encoding)
        result=face_recognition.compare_faces([decoded], unknown_face_encoding)
        face_distances = face_recognition.face_distance([decoded], unknown_face_encoding)
        vale = face_distances
        if np.any(face_distances <= 0.6):
            best_match_idx = np.argmin(face_distances)
            print(best_match_idx)
            d = json.dumps({"userId": id, "code": 0, "faceStatus": "Face Matched", "value": str(vale)})
        else:
            name = None
            d = json.dumps({"userId": id, "code": 1, "faceStatus": "Face Not Matched", "value": str(vale)})
        #face_match_percentage = (1 - face_distances) * 100
        #for i, face_distance in enumerate(face_distances):
        #    global vale
        #    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
        #    print("- comparing with a tolerance of 0.6? {}".format(face_distance < 0.6))
        #    vale = np.round(face_match_percentage, 4)
        #    print(vale)  # upto 4 decimal places
        #if result[0] == True:
        #   d = json.dumps({"userId": id,  "code": 1, "faceStatus": "Face Matched", "value": str(vale)})
        #    print(d)
        #else:
        #    d = json.dumps({"userId": id,  "code": 1, "faceStatus": "Face Not Matched", "value": str(vale)})
        #    print(d)
    except Exception as e:
        print(e)
        d = json.dumps({"userId": id, "code": 3, "faceStatus": "Face NOT Detected","value": str(vale)})
    #url = 'http://127.0.0.1:8000/'
    #d=json.dumps({"userId": id, "faceStatus": result2})
    print(d)
    #f.write(d + '\n')
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    re = requests.post(url, data=d, verify=False)
    print(re)
    return d


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5009, debug=True)
