import torch
try:
    model = torch.jit.load('models/yolopv2.pt')
    print("It is a TorchScript file!")
except Exception as e:
    print(f"Not TorchScript: {e}")
    try:
        ckpt = torch.load('models/yolopv2.pt')
        print(f"It is a PyTorch checkpoint. Keys: {ckpt.keys() if isinstance(ckpt, dict) else 'Not a dict'}")
    except Exception as e2:
        print(f"Not a checkpoint either: {e2}")
