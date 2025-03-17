# Image Classification API

## Overview
This project builds and deploys an image classification model using ResNet18 trained on the CIFAR-10 dataset. The trained model is exposed as a FastAPI-based REST API.


## Features
- Preprocesses CIFAR-10 dataset (resize, normalize, augment)
- Trains a ResNet18 model using PyTorch
- Deploys model via FastAPI
- Provides an API endpoint for image classification



## Requirements
Make sure you have Python 3.8+ installed. Install dependencies using:
```bash
pip install torch torchvision fastapi uvicorn pillow numpy requests
```



## Running the Project

### 1. Train the Model
Run the following command to train the model and save it:
```bash
python your_script.py
```
This will preprocess the dataset, train the model, and save it as `model.pth`.



### 2. Start the FastAPI Server
To launch the API, run:
```bash
uvicorn your_script:app --host 0.0.0.0 --port 8000
```
Replace `your_script` with your actual filename (e.g., `main`).



### 3. Test the API
Use **cURL** to send an image for classification:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@image.jpg'
```
Alternatively, use **Postman**:
1. Set method to `POST`
2. URL: `http://127.0.0.1:8000/predict`
3. In Body -> form-data, upload an image under the `file` key.



### Expected Response:
```json
{
  "prediction": 3
}
```
The number represents a class in CIFAR-10 (e.g., cat, dog, etc.).



## Deployment with Docker
You can containerize the API using Docker:
```dockerfile
FROM python:3.8
WORKDIR /app
COPY . .
RUN pip install torch torchvision fastapi uvicorn pillow numpy
CMD ["uvicorn", "your_script:app", "--host", "0.0.0.0", "--port", "8000"]
```


Then build and run:
```bash
docker build -t image-classifier .
docker run -p 8000:8000 image-classifier
```

## Notes
- Ensure the dataset downloads correctly by checking the `./data` directory.
- Modify `your_script.py` for custom models or datasets.



## Author
Nishkarsh Kumar

