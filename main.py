from fastapi import FastAPI, File, UploadFile
from gradio_client import Client, handle_file

app = FastAPI()
client = Client("shahad-alh/Arabi_char_classifier")

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_path = f"temp_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Send image to Hugging Face Space
    try:
        prediction = client.predict(handle_file(temp_path), api_name="/predict")
    except Exception as e:
        return {"error": str(e)}

    return {"prediction": prediction}
    
@app.get("/")
def root():
    return {"message": "Batoot is here!!"}
