
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

# def encode_vector(ar):
#     return base64.encodestring(ar.tobytes()).decode('ascii')
# def decode_vector(ar):
#     return np.fromstring(base64.decodestring(bytes(ar.decode('ascii'), 'ascii')), dtype='uint8')

r = redis.Redis(host='localhost', port=6379, db=0)
url = "https://leap-uat-bucket.s3.ap-south-1.amazonaws.com/avatars/original/13107.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWV4WRNEORWP6HLWO/20210302/ap-south-1/s3/aws4_request&X-Amz-Date=20210302T084519Z&X-Amz-Expires=86400&X-Amz-Signature=c3edeabfb543f279f6a47ff7f097d39d1363b2629766520bada97fade91280aa&X-Amz-SignedHeaders=host"
logo1 = imutils.url_to_image(url)
print(type(logo1))
shape=logo1.shape
# d_orig_packed = m.packb(logo1)
#r.set("13107", encoded)

# Set the data in redis
r.set('13107', logo1.tobytes())
d_out = r.get('13107')
print(shape)
image = np.fromstring(d_out, dtype=np.uint8)
decoded=image.reshape(shape)#(2320, 2320, 3)
url = "https://leap-uat-bucket.s3.ap-south-1.amazonaws.com/avatars/original/13107.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWV4WRNEORWP6HLWO/20210302/ap-south-1/s3/aws4_request&X-Amz-Date=20210302T084519Z&X-Amz-Expires=86400&X-Amz-Signature=c3edeabfb543f279f6a47ff7f097d39d1363b2629766520bada97fade91280aa&X-Amz-SignedHeaders=host"
logo = imutils.url_to_image(url)
print(type(logo))
my_face_encoding = face_recognition.face_encodings(decoded)[0]
unknown_face_encoding = face_recognition.face_encodings(logo)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

print("sss")
print(results)
# print(decoded)


# Retrieve and unpack the data

# Check they match
# assert np.alltrue(logo1 == d_out)
# assert logo1.dtype == d_out.dtype



# r.flushdb()
#url = "https://leap-uat-bucket.s3.ap-south-1.amazonaws.com/avatars/original/13107.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWV4WRNEORWP6HLWO/20210302/ap-south-1/s3/aws4_request&X-Amz-Date=20210302T084519Z&X-Amz-Expires=86400&X-Amz-Signature=c3edeabfb543f279f6a47ff7f097d39d1363b2629766520bada97fade91280aa&X-Amz-SignedHeaders=host"
#logo1 = imutils.url_to_image(url)
#encoded=encode_vector(logo1)
#r.set("13107", encoded)
# dat=decode_vector(r.get("13107"))
# print(dat)

# Yes, flushdb() and flushall() both exist.
# import redis
# r = redis.Redis()
#r.flushdb()
