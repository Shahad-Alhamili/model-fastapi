from fastapi import FastAPI, File, UploadFile
from gradio_client import Client, handle_file

app = FastAPI()
client = Client("shahad-alh/Arabi_char_classifier")

@app.post("/classify")
def classify_image(file: UploadFile = File(...)):
    # Saving the uploaded image temporarily
    temp_path = f"temp_{file.filename}"

    # Send image to HF Space
    try:
        prediction = client.predict(handle_file(temp_path), api_name="/predict")
    except Exception as e:
        return {"error": str(e)}

    return {"prediction": prediction}
    
@app.get("/")
def root():
    return {"message": "Batoot is here!!"}
