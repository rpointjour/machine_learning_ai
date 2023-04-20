# PyTorch
<img src="https://user-images.githubusercontent.com/54840122/233464516-8a8b2be0-a533-49ab-9e2a-6aa579bac553.png"
style="width:25%;height:25%;">

<i>reference: <a>https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html</a></i>

- To use PyTorch: Install & import module
  - `import torch`
- **Tensors**
  - Multi-dimensional arrays
  - Ways to initialize tensors
    1. As zeros:
    
      `z = torch.zeros(5,3)`
    
    2. As ones:
    
      `z = torch.ones(5,3)`
    
    3. Random: To initialize learning weights!
       
       `torch.manual_seed(1732)`
       
       `r1 = torch.rand(3,3)`
       
       `r2 = torch.rand(3,3)`
       
       `torch.manual_seed(1732)`
       
       `r3 = torch.rand(3,3)`
       
- **Models**
- **Datasets & Dataloaders**
- **Training Models**
- **Research Prototyping**
- **Production Deployment**
