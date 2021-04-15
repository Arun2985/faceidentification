import json
from fastapi import BackgroundTasks, FastAPI
import redis
import numpy as np
import uvicorn
from typing import Optional
from pydantic import BaseModel
import face_recognition
import imutils
import asyncio
import asyncio
import time
import requests
from concurrent.futures import ThreadPoolExecutor


_executor = ThreadPoolExecutor(1)
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
@app.post("/items")
async def create_item(item: Item):
    id = item.Emp_id
    Img = item.Image_url
    print(Img)
    logo2 = imutils.url_to_image(Img)
    logo2 = imutils.resize(logo2, width=480)
    # img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    ##img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
    # img1 = img1[:, :, ::-1]
    # img2 = img2[:, :, ::-1]
    unknown_face_encoding = face_recognition.face_encodings(logo2)[0]
    E_id = r.get(id)
    image = np.fromstring(E_id, dtype=np.float)
    decoded = image.reshape((128,))
    dist = np.linalg.norm(decoded-unknown_face_encoding)
    result = {}
    if dist < 0.45:
        url = ' http://127.0.0.1:5000/'
        result = {"message": str(dist)}
        x = await loop.run_in_executor(_executor, requests.post(url, data=result))
        print(x)
    else:
        url = ' http://127.0.0.1:5000/'
        result = {"message": str(dist)}
        x =  await loop.run_in_executor(_executor,requests.post(url, data=result))
        print(x)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(create_item())
    loop.close()