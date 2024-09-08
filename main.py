from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import pytesseract
from PIL import Image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os

# تحديث مسار tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكنك تحديد نطاقات معينة بدلاً من "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # تحويل الصورة من BGR إلى RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # تحويل الصورة إلى PIL Image
    pil_image = Image.fromarray(frame_rgb)
    # استخراج النصوص من الصورة
    text = pytesseract.image_to_string(pil_image, lang='eng')
    response = {"text": text}
    return JSONResponse(content=response)

# تأكد من ضبط متغير البيئة GEMINI_API_KEY بالمفتاح الصحيح
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

@app.post("/generate-content/")
async def generate_content(prompt: str = Form(...), image: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await image.read()))

        # افتراض أنك تقوم بحفظ الصورة مؤقتاً لاستخدامها مع النموذج
        img.save("temp_image.jpg")
        img = Image.open("temp_image.jpg")

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt, img])
        return JSONResponse(content={"text": response.text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
