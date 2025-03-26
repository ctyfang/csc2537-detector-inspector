import torch

import torch
import numpy as np
import pandas as pd

def torch_array_to_csv(tensor, filename="tensor_data.csv"):
    """
    Convert a PyTorch tensor to a CSV file.
    
    Parameters:
    tensor (torch.Tensor): The PyTorch tensor to convert
    filename (str): The name of the CSV file to create
    
    Returns:
    str: Path to the saved CSV file
    """
    # Convert tensor to numpy array
    if tensor.requires_grad:
        numpy_array = tensor.detach().cpu().numpy()
    else:
        numpy_array = tensor.cpu().numpy()
    
    # Handle different dimensions
    if len(numpy_array.shape) == 1:
        # 1D tensor
        df = pd.DataFrame(numpy_array)
    elif len(numpy_array.shape) == 2:
        # 2D tensor (already in matrix form)
        df = pd.DataFrame(numpy_array)
    else:
        # Higher-dimensional tensors need to be reshaped
        # This is a simple approach - just flatten the tensor
        df = pd.DataFrame(numpy_array.reshape(numpy_array.shape[0], -1))
        
    # Save to CSV
    df.to_csv(filename, index=False, header=False)
    
    return filename

filename = "0b5142c1-420b-3fea-9e98-b87327ae22c6_315968127249927215"

intrinsics = torch.load(f"data/calibration/{filename}_intrinsics.pt")
extrinsics = torch.load(f"data/calibration/{filename}_extrinsics.pt")
torch_array_to_csv(intrinsics.data.view(4,4)[:3, :3], f"intrinsics.csv")
torch_array_to_csv(extrinsics.data.view(4,4), f"extrinsics.csv")