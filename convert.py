from RepCFT import *

model = build(num_classes=8)
weights = torch.load('./RepCFT_training.pth', map_location='cpu')
model.load_state_dict(weights)

model2 = RepLRNet_model_convert(model, './RepCFT_deploy.pth')
print(model2)