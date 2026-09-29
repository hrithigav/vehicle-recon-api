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
        
        # 1. Convert to normalized tensor FIRST (matching training environment)
        tensor_in = torch.from_numpy(roi).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0
        
        # 2. Inject occlusion using the exact scalar (0.0) the network learned to inpaint
        tensor_in[:, :, 80:160, 80:160] = 0.0
        
        with torch.no_grad(): out = gan_model(tensor_in)
        
        # 3. Denormalize tensors back to BGR for UI transmission
        damaged_np = ((tensor_in.squeeze().permute(1,2,0).numpy() + 1.0) * 127.5).astype(np.uint8)
        out_img = ((out.squeeze().permute(1,2,0).numpy() + 1.0) * 127.5).astype(np.uint8)
        
        return {
            'status': 'success', 
            'confidence': float(res[0].boxes[0].conf[0]),
            'damaged_image_base64': base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(damaged_np, cv2.COLOR_RGB2BGR))[1]).decode('utf-8'),
            'reconstructed_image_base64': base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))[1]).decode('utf-8')
        }
    except Exception as e: raise HTTPException(500, str(e))
    finally: gc.collect()
