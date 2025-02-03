import torch 

T =torch.rand(5, requires_grad=True)

X = T**2
Y=X**2
Z = Y.mean() 
Y.retain_grad()
X.retain_grad() 

print("Grad before backpropagation")
print(f"Y = {Y}")
print(f"grad at Y = {Y.grad}") 
print(f"X = {X}")
print(f"grad at X = {X.grad}")
print(f"T = {T}")
print(f"grad at T = {T.grad}") 

Z.backward() 

print("###############################")
print("Grad after backpropagation")
print(f"Y = {Y}")
print(f"grad at Y = {Y.grad}") 
print(f"X = {X}")
print(f"grad at X = {X.grad}")
print(f"T = {T}")
print(f"grad at T = {T.grad}") 
