import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
    
    # Test tensor on GPU
    x = torch.tensor([1.0, 2.0]).cuda()
    print("GPU test tensor:", x.device)