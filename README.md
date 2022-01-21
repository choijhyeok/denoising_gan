# GAN Denoising

&nbsp;

## 1. 목적 및 동기

 - CT, MRI 이미지의 큰 차이점중에서 잡음이 많이 제거된다는 특징이 존재함
 - 실제로 CT 이미지의 잡음을 보다 잘 제거한다면 MRI를 대체할수 있지 않을까 라는 아이디어에서 수행
 - salt & paper , Gaussian, speckle 잡음을 통해서 이미지에 잡음 추가
 - FID score를 통해서 최종적인 score 확인
 
&nbsp;

## 2. 데이터
 - CT dataset : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    - CT 이미지 셋


&nbsp;
## 3. GAN 란?

 - 잔차제곱합을 최소화 하는 가중치 벡터를 구하는 방법
 - OLS 사용하면 독립변수 x에 대해 종속변수 y에 영향이 있는지 확인가능 , 또한 P-value도 확인가능
 - 회귀 방정식을 구할수 있고, 각각의 계수를 확인할수 있음
 
&nbsp;
## 4. 전체적인 구성

