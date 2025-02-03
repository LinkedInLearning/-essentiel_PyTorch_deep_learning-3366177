import pandas as pd
import torch

df = pd.read_csv('tensor_vs_pandas.csv') 
print(df) 

tensor = torch.tensor(df.values) 
print("Tensor : ")
print(tensor) 
print(f"Shape : {tensor.shape}") 
