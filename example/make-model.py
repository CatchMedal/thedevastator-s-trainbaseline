import torch
import ssl
from efficientnet_pytorch import EfficientNet
# from torchsummaryX import summary
from torchinfo import summary
ssl._create_default_https_context = ssl._create_unverified_context

resnext50 = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
ef_b4 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
#ef_b7 = EfficientNet.from_pretrained('efficientnet-b7')

# summary(resnext50,input_size=(64,3,256,256))
# summary(ef_b4,input_size=(64,3,256,256))
# summary(ef_b4,input_size=(64,3,256,256))

print(resnext50)