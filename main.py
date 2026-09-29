import base64, gc, os, cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['POST'])

yolo_model = YOLO('yolov8n.pt')

@app.post('/reconstruct')
async def reconstruct(file: UploadFile = File(...)):
    try:
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = yolo_model.predict(img_rgb, classes=[2,3,5,7], verbose=False)
        if not res[0].boxes: raise HTTPException(404, 'No target vehicle detected.')
        
        box = res[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        roi = cv2.resize(img_rgb[box[1]:box[3], box[0]:box[2]], (256, 256))
        
        # 1. Generate precise binary defect mask
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        mask[80:160, 80:160] = 255
        
        # 2. Inject structural damage
        damaged_roi = roi.copy()
        damaged_roi[mask == 255] = 0
        
        # 3. Execute Navier-Stokes fluid dynamics reconstruction
        reconstructed_roi = cv2.inpaint(damaged_roi, mask, 3, cv2.INPAINT_NS)
        
        # Convert BGR for transmission
        dmg_img_bgr = cv2.cvtColor(damaged_roi, cv2.COLOR_RGB2BGR)
        out_img_bgr = cv2.cvtColor(reconstructed_roi, cv2.COLOR_RGB2BGR)
        
        return {
            'status': 'success', 
            'confidence': float(res[0].boxes[0].conf[0]),
            'damaged_image_base64': base64.b64encode(cv2.imencode('.jpg', dmg_img_bgr)[1]).decode('utf-8'),
            'reconstructed_image_base64': base64.b64encode(cv2.imencode('.jpg', out_img_bgr)[1]).decode('utf-8')
        }
    except Exception as e: raise HTTPException(500, str(e))
    finally: gc.collect()
