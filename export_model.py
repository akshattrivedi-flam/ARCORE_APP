import torch
import torch.nn as nn
from torchvision import models
import os
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

# --- MODEL DEFINITION (Must match training) ---
class ObjectronMobileNetV3(nn.Module):
    def __init__(self, num_keypoints=9):
        super(ObjectronMobileNetV3, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights=None) 
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1024, 512),
            nn.Hardswish(inplace=True),
            nn.Linear(512, num_keypoints * 2),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.backbone(x)

def export():
    MODEL_PATH = "bottle_objectron_overfit.pth"
    ONNX_PATH = "bottle_objectron.onnx"
    TF_PATH = "bottle_tf_saved_model"
    TFLITE_PATH = "bottle_objectron.tflite"
    
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    model = ObjectronMobileNetV3()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # Dummy Input (Batch size 1, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting to ONNX: {ONNX_PATH}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_PATH,
        opset_version=12,
        input_names=['input'],
        output_names=['keypoints'],
        dynamic_axes={} # Static shape for TFLite
    )
    print("ONNX export complete.")

    try:
        print("Loading ONNX model...")
        onnx_model = onnx.load(ONNX_PATH)
        
        print("Converting ONNX to TensorFlow SavedModel...")
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(TF_PATH)
        print(f"SavedModel written to {TF_PATH}")
        
        print("Converting SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
        # Optimizations for mobile (quantization optional)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
            
        print(f"SUCCESS: TFLite model generated at {TFLITE_PATH}")

    except Exception as e:
        print(f"Error during ONNX->TF->TFLite conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export()
