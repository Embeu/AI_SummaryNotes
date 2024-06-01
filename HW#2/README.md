# 인공지능개론 과제 #2

### 201904022 김상옥, 202284026 안정빈

----
#### 내용 : Fashion MNIST 데이터셋을 이용한 CNN 이미지 분류

#### 소스코드 : CNN 소스코드 -  SimpleConvNet (p251 ~ p253 ) 만을 이용해야 함. (수업 시간에 실습 진행함)
#### 데이터셋 : [패션 MNIST](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)

#### 조건 : 하이퍼파라미터 튜닝과 CNN 신경망 구조 변경 관련한 제한 조건은 없음.

----

### 소스코드에서 튜닝한 값

- ##### 배치 사이즈 : 100
- ##### hidden_size : 100
- ##### 계층 : COV-배치정규화-ReLU-Pool-Affine-배치정규화-ReLU-Affine-SoftmaxWithLoss
- ##### learning_rate(학습률) : 0.005


### 패션 MNIST 불러오기
```python
import gzip
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 파일에서 데이터를 읽어 들입니다.
        data = f.read()
        # 이미지는 16바이트 헤더를 가지고 있으므로, 이를 건너뛰고 나머지 데이터를 읽어옵니다.
        images = np.frombuffer(data, np.uint8, offset=16)
        # 이미지를 (num_samples, 28, 28) 형식으로 재구성합니다.
        images = images.reshape(-1, 1, 28, 28)
    return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # 파일에서 데이터를 읽어 들입니다.
        data = f.read()
        # 레이블은 8바이트 헤더를 가지고 있으므로, 이를 건너뛰고 나머지 데이터를 읽어옵니다.
        labels = np.frombuffer(data, np.uint8, offset=8)
    return labels

# 파일 경로를 설정합니다.
train_images_path = 'Fashion-MNIST/train-images-idx3-ubyte.gz'
train_labels_path = 'Fashion-MNIST/train-labels-idx1-ubyte.gz'
test_images_path = 'Fashion-MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = 'Fashion-MNIST/t10k-labels-idx1-ubyte.gz'
```
  
### 사용한 함수
- ##### Affine (계층 클래스)
- ##### ReLu (활성화 함수 클래스)
- ##### softmax (출력층 활성화 함)
- ##### cross_entropy_error (교차 엔트로피 오차 / 손실 함수)
- ##### numerical_gradient (수치 미분 및 기울)
- ##### SoftmaxWithLoss (활성화 함수 클래)
- ##### AdaGrad (최적화 함수)
- ##### He_init (ReLu 가중치 초기화 함수)
- ##### 배치 정규화 (예제 소스 활용)

### CNN 함수, 클래스
- SimpleConvNet (CNN 계층 클래스)
- Convolution (합성곱 계층 클래스)
- Pooling (풀링 클래스)
- im2col (n차원 -> 2차원)
- col2im (2차원 -> n차)
