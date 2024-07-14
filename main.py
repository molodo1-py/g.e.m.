from keras.models import load_model
from tensorflow import random
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import io
import base64
from PIL import Image, ImageOps
import numpy as np
from typing import Dict
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from DCGAN.generator import generator_model


app = FastAPI()
app.mount("/static", StaticFiles(directory=f'{os.getcwd()}/static'), name="static")
templates = Jinja2Templates(directory="templates")

#Запросы CORS от всех источников
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Загружаем модели
model_mnist = load_model(f"{os.getcwd()}/weights/MnistConv2d.h5")
model_anime = generator_model()
model_anime.load_weights(f"{os.getcwd()}/weights/anime_faces.h5")

#Запросы/ответы
@app.get("/")
@app.get("/index.html", response_class=HTMLResponse)
async def home(request: Request):
    '''Домашняя страница -> кликер'''
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/mnist.html", response_class=HTMLResponse)
async def mnist(request: Request):
    return templates.TemplateResponse("mnist.html", {"request": request})

@app.get("/anime.html", response_class=HTMLResponse)
async def anime(request: Request):
    return templates.TemplateResponse("anime.html", {"request": request})

@app.post("/mnist_predict")
async def predict(data: Dict[str,str]):
    '''Кнопка <РАСПОЗНАТЬ>'''
    image_data = base64.b64decode(data['image_data'])
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('L').resize((28,28))
    image = np.array(image)
    processed_image = np.expand_dims(image, axis=0)
    predictions = model_mnist.predict(processed_image)
    digit = np.argmax(predictions)
    return {"digit": int(digit)}

@app.get("/anime_generate")
async def generate(request: Request):
    '''Кнопка <СГЕНЕРИРОВАТЬ АВАТАР>'''
    seed = random.normal([1, 100])
    generate = model_anime(seed, training=False)
    generate = (generate[0].numpy() + 1)/2
    new_img = Image.fromarray((generate * 256).astype(np.uint8)).convert("RGB")
    buffer_inverted = io.BytesIO()
    new_img.save(buffer_inverted, format="PNG")
    buffer_inverted.seek(0)
    
    return StreamingResponse(buffer_inverted, media_type="image/png")

#Запуск приложения
if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get('PORT', 8000)))