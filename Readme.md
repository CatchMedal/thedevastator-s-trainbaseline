https://www.kaggle.com/code/thedevastator/training-fastai-baseline

# 모델

![img](https://user-images.githubusercontent.com/30853787/186518724-9df184f7-9d14-4d1f-9653-0b02bd25069f.jpg)

UNext50의 backbone인 ResNet50을 EfficientNet_b4로 전환, 다이어그램에서 FPN은 생략되었습니다.



# Kaggle Inference Submission 이슈 

* 결과 제출을 위한 Kaggle Inference 노트북에서는 인터넷을 사용할 수 없음.

baseline이 ResNet인 경우, torchvision.models에서 import하면 되지만, efficientnet으로 변환하는 과정에서 이미 ndivia걸로 구성해놓아서, inference에서 모델을 불러와야하는 상황!!!

```python
m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')

from torchvision.models.resnet import ResNet


torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
# nvidia 모델은 torchvision에서 불러올 수 없음..
```  
[https://github.com/NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) 이 소스코드를 저장해놓은 데이터셋을 불러와서, 따로 import해주었음

```python
import sys
sys.path.append('/kaggle/input/torchhub-efficientnet/DeepLearningExamples-torchhub/PyTorch')
from Classification.ConvNets.image_classification.models.efficientnet import efficientnet_b4
```

### Solution

model.save() 코드에서, 전체 모델을 저장해주기

```python
#Save
torch.save(learn.model.state_dict(),f'model_{fold}.pth')


#Load state dict
model = EfficientNet.from_pretrained() # !! Inference에서 인터넷 사용 안되므로, 불러오기 실패
model.load_state_dict() #(실패)
```

```python
torch.save(learn.model,f'model_{fold}.pth')
torch.load(path)
```
# Mac 에서 돌려보기
* MPS란? Cuda, 와 같은 선상의 개념 https://dong-life.tistory.com/112
### 맥북 덮고 사용
https://ssumer.com/%ED%84%B0%EB%AF%B8%EB%84%90-%EB%AA%85%EB%A0%B9-%EB%9A%9C%EA%BB%91-%EB%8B%AB%EA%B3%A0-%EB%A7%A5%EB%B6%81-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/
* 잠자기 방지모드 활성  sudo pmset  disablesleep 1 (-c옵션 넣으면 전원 연결된경우만 잠들지 않음)
* 잠자기 방지모드 해제 sudo pmset disablesleep 0
### 속도

훈련데이터 2791개  
fold 4 , batch_size 64
1 fold당 8시간..??
## Error Handling


* torch hub에서 모델 가져올 때 발생하는 ssl 관련 에러  
**<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:997)>**  
아래와 같이 추가
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### GPU 이슈
* gpu 가속 사용하지 않은 상태에서 num_workers > 0이면, 에러 발생!

### ㅡ
* m1기준 GPU가속 : 2022년 5월(따끈..) 배포된, 12.3 + 버전에서 지원함. > 맥북 업데이트중..
https://discuss.pytorch.kr/t/apple-m1-gpu/286/2

* FastAI 모듈 mps디바이스 지원 하지 않는 에러

export PYTORCH_ENABLE_MPS_FALLBACK=1

