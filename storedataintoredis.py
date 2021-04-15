import base64
import pickle
import struct
import cv2
import numpy as np
import redis
import imutils
from PIL import Image
import io
import msgpack as m
import face_recognition
import imutils


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


id="606c4c979df1cc119e8b50fd"
org_img="13276.jpg"
r = redis.Redis(host='localhost', port=6379, db=0)

#r.flushdb()
#url="https://leap-uat-bucket.s3.ap-south-1.amazonaws.com/avatars/original/13047.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWV4WRNEORWP6HLWO%2F20210405%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20210405T083800Z&X-Amz-Expires=1800&X-Amz-Signature=f7069b34f3a6d959caeb32681df29ea4ac68f797c51e686612f46bb43a6b97f8&X-Amz-SignedHeaders=host"
#logo1 = imutils.url_to_image(url)
#print(type(logo1))
# d_orig_packed = m.packb(logo1)
#r.set("13107", encoded)

logo1 = face_recognition.load_image_file(org_img)
locations, encodings = get_face_embeddings_from_image(logo1)
my_face_encoding = encodings[0]
shape=my_face_encoding.shape
print(shape)

# Set the data in redis
r.set(id, my_face_encoding.tobytes())
d_out = r.get(id)
image = np.fromstring(d_out, dtype=np.float)
decoded = image.reshape((128,))
#(2320, 2320, 3)
print(decoded)
