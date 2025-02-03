import torch 
import numpy as np 

'''
A = torch.rand(5) 
print(f"A = {A}") 

B = A.numpy() 
print(f"B = {B}") 
print(f"type(B) = {type(B)}") 

B[0] = 99 
print(f"A = {A}") 
'''
A = np.ones(5) 
B = torch.from_numpy(A)
C = torch.tensor(A) 

print(f"A = {A}")
print(f"B = {B}")
print(f"C = {C}")
A*=10 
print("====================")
print(f"A = {A}")
print(f"B = {B}")
print(f"C = {C}")

