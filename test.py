import asyncio
import json
import time

import cv2
import face_recognition
import imutils
import redis
import numpy as np
import requests
from flask import Flask, request
import os

#app = Flask(__name__)
#r = redis.Redis(host='localhost', port=6379, db=0)
#url = "https://172.17.98.55/api/v2/agent-monitoring/face-status/"
#headers = {
##    'Content-Type': 'application/json'
#}


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


id ="351060"
entries = os.listdir(''+id+'/')
f = open(id+"Arun.txt", "a")
original_pic=id+".jpg"
vale = ""


try:
    #print(data)
    #print(data["userId"])
    #print(data['userId'])
    for entry in entries:
        print(entry)
        #logo1 = imutils.url_to_image(Img)
        #print(logo2)
        #logo2 = imutils.resize(logo1, width=480)
        #print(logo2)
        d=os.getcwd()+"/"+id+"/"+entry
        logo2 = face_recognition.load_image_file(d)
        logo1 = face_recognition.load_image_file(original_pic)
        #dist = np.linalg.norm(decoded - unknown_face_encoding)
        try:
            locations, encodings = get_face_embeddings_from_image(logo2)
            unknown_face_encoding = encodings[0]
            locations, encodings = get_face_embeddings_from_image(logo1)
            decoded= encodings[0]
            #E_id = r.get(id)
            #image = np.fromstring(E_id, dtype=np.float)
            #decoded = image.reshape((128,))
            result=face_recognition.compare_faces([decoded], unknown_face_encoding)
            face_distances = face_recognition.face_distance([decoded], unknown_face_encoding)
            print(face_distances)
            vale=face_distances
            if np.any(face_distances <=0.6):
                best_match_idx = np.argmin(face_distances)
                print(best_match_idx)
                d = json.dumps({"userId": id, "code": entry, "faceStatus": "Face Matched", "value": str(vale)})
            else:
                name = None
                d = json.dumps({"userId": id, "code": 1, "faceStatus": "Face Not Matched", "value": str(vale)})
            #face_match_percentage = (1 - face_distances) * 100
            #for i, face_distance in enumerate(face_distances):
                #print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
                #print("- comparing with a tolerance of 0.6? {}".format(face_distance < 0.6))
            #    vale = np.round(face_match_percentage, 4)
                #print(vale)  # upto 4 decimal places
            #if result[0] == True:
            #    d = json.dumps({"userId": id, "code": entry, "faceStatus": "Face Matched", "value": str(vale)})
            #    #print(d)
            #else:
            #    d = json.dumps({"userId": id, "code": 1, "faceStatus": "Face Not Matched", "value": str(vale)})
            #    #print(d)
            f.write(d + '\n')
        except:
            d = json.dumps({"userId": id, "code": entry, "faceStatus": "Face NOT Detected"})
            f.write(d + '\n')
            continue
except Exception as e:
    print(e)
print(d)
f.close()



#if __name__ == "__main__":
#   app.run(host="0.0.0.0", port=5009, debug=True)
