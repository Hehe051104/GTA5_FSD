import torch
import time
import os
import cv2
import numpy as np

def check_environment():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Current Device: {torch.cuda.current_device()}")
    else:
        print("WARNING: CUDA is NOT available. Running on CPU.")

def check_model_loading(model_path):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print(f"\nLoading model from {model_path}...")
    try:
        start_time = time.time()
        model = torch.jit.load(model_path)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        print(f"Model loaded successfully in {time.time() - start_time:.4f} seconds.")
        
        # Dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            
        print("Running warmup inference...")
        with torch.no_grad():
            model(dummy_input)
            
        print("Running benchmark inference (10 iterations)...")
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                model(dummy_input)
        avg_time = (time.time() - start_time) / 10
        print(f"Average inference time: {avg_time:.4f} seconds ({1/avg_time:.2f} FPS)")
        
    except Exception as e:
        print(f"Error loading or running model: {e}")

if __name__ == "__main__":
    check_environment()
    check_model_loading("models/yolopv2.pt")
