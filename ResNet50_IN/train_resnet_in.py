import torchvision

resnet50_bn = torchvision.models.resnet50(pretrained=True, progress=True)

print(resnet50_bn)