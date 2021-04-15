import json

from fastapi import FastAPI
import redis
import numpy as np
import uvicorn
from typing import Optional
from pydantic import BaseModel
import face_recognition
import imutils

app = FastAPI()
# r = redis.Redis(host='localhost', port=6379, db=0)
# d_out = r.get("13107")
# image = np.fromstring(d_out, dtype=np.uint8)
# decoded=image.reshape((2320, 2320, 3))
# item inherit from basemodel
class Item(BaseModel):
    Emp_id: int
    Image_url: Optional[str] = None

r = redis.Redis(host='localhost', port=6379, db=0)
#k = 1
@app.post("/items")
def create_item(item: Item):
    #global k
    id = item.Emp_id
    #print(k)
    #k += 1
    #result=[]
    Img = item.Image_url
    # print("downloading image")
    print(Img)

    logo2 = imutils.url_to_image(Img)
    logo2 = imutils.resize(logo2, width=480)
    # print(" image encoding")
    unknown_face_encoding =  face_recognition.face_encodings(logo2)[0]
    # print("redis connection established & get data by id")
    E_id = r.get(id)
    # print("decoded bytes into ndarray")
    image = np.fromstring(E_id, dtype=np.float)
    decoded = image.reshape((128,))
    # print(" encoding of redis image")
    #my_face_encoding = face_recognition.face_encodings(decoded)[0]
    # print("comparing the image")
    # results = face_recognition.compare_faces([decoded], unknown_face_encoding)
    # # print(results)
    # if results[0] == True:
    #np.linalg.norm(face_encodings - face_to_compare, axis=1)
    dist = np.linalg.norm(decoded-unknown_face_encoding)
    # print(dist)
    # result = {}
    if dist < 0.45:
        result = {"code": 3, "message": dist}
        return json.dumps(result)
        #print("It's a picture of me!")
        #return {"code": 1, "message": "Face Matched"}
    else:
        #print("It's not a picture of me!")
        result = {"code": 3, "message": str(list)}
        return json.dumps(result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)