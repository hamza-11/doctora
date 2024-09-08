from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
from PIL import Image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكنك تحديد نطاقات معينة بدلاً من "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تأكد من ضبط متغير البيئة GEMINI_API_KEY بالمفتاح الصحيح
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # تحويل الصورة من BGR إلى RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # تحويل الصورة إلى PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    # استخدام Gemini للتعرف على النص في الصورة
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = "Extract and return only the text from this image. Do not include any additional commentary or description."
    response = model.generate_content([prompt, pil_image])
    
    return JSONResponse(content={"text": response.text})

@app.post("/generate-content/")
async def generate_content(prompt: str = Form(...), image: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await image.read()))
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt, img])
        return JSONResponse(content={"text": response.text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
