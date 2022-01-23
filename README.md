# GAN Denoising

&nbsp;

## 1. 목적 및 동기

 - CT, MRI 이미지의 큰 차이점 중에서 잡음이 많이 제거된다는 특징이 존재함
 - 실제로 CT 이미지의 잡음을 더욱 잘 제거한다면 MRI를 대체할 수 있지 않을까 라는 아이디어에서 수행
 - salt & pepper , Gaussian, speckle 잡음을 통해서 이미지에 잡음 추가
 - FID score를 통해서 최종적인 score 확인
 
&nbsp;

## 2. 데이터
 - CT dataset : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    - CT 이미지 셋


&nbsp;
## 3. GAN 란?
![gan이미지](http://drive.google.com/uc?export=view&id=1GyK-KOLMmZiXL63YmxkVxmTYtd6A7AS7)


 - 정의 : 생성자와 식별자가 서로 경쟁(Adversarial)하며 데이터를 생성(Generative)하는 모델(Network)을 뜻합니다.
 - Generator(생성자) : 생성된 z를 받아 실제 데이터와 비슷한 데이터를 만들어내도록 학습
 - Discriminator(구분자) : 실제 데이터와 생성자가 생성한 가짜 데이터를 구별하도록 학습

&nbsp;

## 4. GAN 과정
 1. GAN의 훈련과정은 이미지 크기만큼의 z 잡음벡터가 Generator에 입력
 2. Generator에서 Fake 이미지 생성, 실제 이미지와 같이 Discriminator에 입력
 3. Discriminator가 Fake인지 Real인지 구분  

&nbsp;
## 5. 변경점
 - Generator에 이미지 크기만큼의 잡음 벡터 z가 입력으로 들어감
 - 이때의 입력을 원본이미지에서 salt & paper , Gaussian, speckle 잡음을 추가한 이미지를 입력으로 전송
 - 최종적으로 Generator가 실제 이미지처럼 변경시키면서 잡음을 제거하게 됨

&nbsp;

## 6. 잡음
 1. salt & pepper
    : 사진 위에 소금과 후추를 떨어트려 놓은 것처럼 보이는 잡음
 2. Gaussian
    : 정규분포를 가지는 잡음, 일반적인 잡음이며 자연 상에서 쉽게 발생하는 잡음
 3. speckle
    : Gaussian 잡음의 불규칙한 값들을 영상에 더하는 잡음에 반해 speckle은 불규칙한 값들을 화소에 곱하는 형태
    
&nbsp;

## 7. FID
 - FID는 생성된 영상의 품질을 평가하는데 사용
 - 이 지표는 영상 집합 사이의 거리를 나타낸다.
 - FID는 GAN을 사용해 생성된 영상의 집합과 실제 생성하고자 하는 클래스 데이터의 분포 거리를 계산한다. 
 - 거리가 가까울수록 좋은 영상으로 판단한다. 
 - Inception 네트워크를 사용하여 원본과 잡음 제거된 이미지 사이의 정규분포를 그리고 두 개의 정규분포 꼭짓점 사이의 거리가 가까울수록 일치율이 높음

&nbsp;

## :heavy_exclamation_mark: 최종적으로 잡음별 loss 그래프로 성능 확인 및 최종은 FID score로 확인!
