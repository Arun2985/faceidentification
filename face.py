import face_recognition
import imutils
from numpy import asarray
from PIL import Image
import numpy as numpy

url = "https://leap-uat-bucket.s3.ap-south-1.amazonaws.com/avatars/original/13107.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAWV4WRNEORWP6HLWO/20210302/ap-south-1/s3/aws4_request&X-Amz-Date=20210302T084519Z&X-Amz-Expires=86400&X-Amz-Signature=c3edeabfb543f279f6a47ff7f097d39d1363b2629766520bada97fade91280aa&X-Amz-SignedHeaders=host"
logo = imutils.url_to_image(url)
print(type(logo))
image2 = Image.fromarray(logo)
print(type(image2))
url2 = "https://i1.wp.com/godofindia.com/wp-content/uploads/2017/05/shahrukh-khan-5.jpg"
logo2 = imutils.url_to_image(url2)
image3 = Image.fromarray(logo2)


#picture_of_me = face_recognition.load_image_file(image2)
my_face_encoding = face_recognition.face_encodings(logo)[0]

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!
#unknown_picture = face_recognition.load_image_file(image3)
unknown_face_encoding = face_recognition.face_encodings(logo2)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
if results[0] == True:
    print(results)
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")

face_distances = face_recognition.face_distance([my_face_encoding], unknown_face_encoding)
face_match_percentage = (1-face_distances)*100
for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))

    print("- comparing with a tolerance of 0.6? {}".format(face_distance < 0.6))
    
    print (numpy.round(face_match_percentage,4)) #upto 4 decimal places
