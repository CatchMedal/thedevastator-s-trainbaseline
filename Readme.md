https://www.kaggle.com/code/thedevastator/training-fastai-baseline

# Info

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

