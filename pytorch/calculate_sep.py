import torch
import torchvision


def sep(w):
    num_class = w.shape[0]
    w = torch.nn.functional.normalize(w, p=2, dim=1)
    cos = torch.mm(w, w.t())
    cos.clamp(-1, 1)
    cos.scatter_(1, torch.arange(num_class).view(-1, 1).long(), -100)
    _, indices = torch.max(cos, dim=0)
    label = torch.zeros((num_class, num_class))
    label.scatter_(1, indices.view(-1, 1).long(), 1) # fill with 1
    return torch.dot(cos.view(cos.numel()), label.view(label.numel())) / num_class
    

resnet18 = torchvision.models.resnet.resnet18(pretrained=True)

for name, p in resnet18.named_parameters():
    if name == "fc.weight":
        fc = p.data

print(fc.shape)
print(fc.min(), fc.max())

print(sep(fc))