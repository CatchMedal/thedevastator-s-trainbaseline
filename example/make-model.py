import torch
import ssl
from efficientnet_pytorch import EfficientNet
from torchvision.models import efficientnet_b4, efficientnet_v2_l, efficientnet_b7
# from torchsummaryX import summary
from torchinfo import summary
ssl._create_default_https_context = ssl._create_unverified_context

#resnext50 = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
# ef_b4 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
#ef_b7 = EfficientNet.from_pretrained('efficientnet-b7')
ef_b4_tv = efficientnet_b4()
ef_v2l_tv = efficientnet_v2_l()
ef_b7_tv = efficientnet_b7()

# summary(resnext50,input_size=(64,3,256,256))
#summary(ef_b4,input_size=(64,3,256,256)) 
# summary(ef_b4,input_size=(64,3,256,256) )
# print(ef_v2l_tv.features[0].Conv2dNormActivation) 


model = torch.load('../models/model_0.pth')
print(model)