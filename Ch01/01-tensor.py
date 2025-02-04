import torch 
#print(f"La version de PyTorch installée est : {torch.__version__}") 
 
a1 = torch.empty(1) 
print(f"a1={a1}") 

a2 = torch.empty(2) 
print(f"a2={a2}")

a3 = torch.empty(3) 
print(f"a3={a3}")

a4 = torch.empty(2,3) 
print(f"a4={a4}")
 

a5 = torch.rand(2,2) 
print(f"a5={a5}")
 

a6 = torch.zeros(3) 
print(f"a6={a6}")

a7 = torch.ones(3)
print(f"a7={a7}")






























