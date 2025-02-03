import torch 

a1 = torch.ones(2,3) 
print(f"a1 = {a1}") 

print(f"Size a1={a1.size()}") 
print(f"Shape a1 ={a1.shape}") 
print(f"a1 data type = {a1.dtype} ") 

a2 = torch.ones(2,3, dtype = torch.float16)
print(f"a2 data type = {a2.dtype}")
 