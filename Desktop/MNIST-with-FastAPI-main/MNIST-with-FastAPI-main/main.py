from fastapi import FastAPI
import uvicorn
from fastapi import UploadFile, File
import model
import numpy as np


app = FastAPI()

@app.get('/predict')
def predict_number(file: UploadFile = File(...)):
    image = model.get_image(np.asarray(file))
    prediction = model.predict(image)

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host='0.0.0.0')