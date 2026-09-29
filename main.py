import base64, gc, os, cv2, numpy as np, torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from model_gan import ResUNetGenerator

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['POST'])

yolo_model = YOLO('yolov8n.pt')
gan_model = ResUNetGenerator()
if os.path.exists('gan_weights.pth'): gan_model.load_state_dict(torch.load('gan_weights.pth', map_location='cpu'))
gan_model = torch.ao.quantization.quantize_dynamic(gan_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
gan_model.eval()

@app.post('/reconstruct')
async def reconstruct(file: UploadFile = File(...)):
    try:
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = yolo_model.predict(img_rgb, classes=[2,3,5,7], verbose=False)
        if not res[0].boxes: raise HTTPException(404, 'No vehicle')
        box = res[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        roi = cv2.resize(img_rgb[box[1]:box[3], box[0]:box[2]], (256, 256))
        tensor_in = torch.from_numpy(roi).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0
        with torch.no_grad(): out = gan_model(tensor_in)
        out_img = cv2.cvtColor(((out.squeeze().permute(1,2,0).numpy() + 1.0) * 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return {'status': 'success', 'confidence': float(res[0].boxes[0].conf[0]), 'reconstructed_image_base64': base64.b64encode(cv2.imencode('.jpg', out_img)[1]).decode('utf-8')}
    except Exception as e: raise HTTPException(500, str(e))
    finally: gc.collect()