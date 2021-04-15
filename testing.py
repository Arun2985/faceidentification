import base64

import face_recognition
import imutils
import redis
from PIL import Image
import numpy as np

#url = "https://leap-uat-bucket.s3.ap-south-1.amazonaws.com/avatars/original/13107.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWV4WRNEORWP6HLWO/20210302/ap-south-1/s3/aws4_request&X-Amz-Date=20210302T084519Z&X-Amz-Expires=86400&X-Amz-Signature=c3edeabfb543f279f6a47ff7f097d39d1363b2629766520bada97fade91280aa&X-Amz-SignedHeaders=host"
url="https://leap-uat-bucket.s3.ap-south-1.amazonaws.com/ankit.lohan%40kochartech.com/01-04-2021/cam-img/original/ankit.lohan%40kochartech.com-2021-04-01T17%3A03%3A10%2B05%3A30.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWV4WRNEORWP6HLWO%2F20210401%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20210401T113312Z&X-Amz-Expires=1800&X-Amz-Signature=33e1c426b9caec7399161286b850cd5e4b84f935fbccaa6ca36b48a5ba730aab&X-Amz-SignedHeaders=host"
logo = imutils.url_to_image(url)
print(type(logo))

my_face_encoding = face_recognition.face_encodings(logo)[0]
print(my_face_encoding.dtype)
print(type(my_face_encoding))
r = redis.Redis(host='localhost', port=6379, db=0)
r.flushall()
# k = r.get("13107")
# l = np.fromstring(k,dtype="float64")


exit()



# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!
# unknown_picture = face_recognition.load_image_file(image3)
unknown_face_encoding = face_recognition.face_encodings(logo2)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
if results[0] == True:
    print(results)
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")

face_distances = face_recognition.face_distance([my_face_encoding], unknown_face_encoding)
face_match_percentage = (1 - face_distances) * 100
for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))

    print("- comparing with a tolerance of 0.6? {}".format(face_distance < 0.6))

    print(np.round(face_match_percentage, 4))  # upto 4 decimal places
