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

<br/>

### 패션 MNIST 불러오기
```python
import gzip

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
        images = np.frombuffer(data, np.uint8, offset=16)
        images = images.reshape(-1, 1, 28, 28)
    return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
        labels = np.frombuffer(data, np.uint8, offset=8)
    return labels

# 파일 경로
train_images_path = 'Fashion-MNIST/train-images-idx3-ubyte.gz'
train_labels_path = 'Fashion-MNIST/train-labels-idx1-ubyte.gz'
test_images_path = 'Fashion-MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = 'Fashion-MNIST/t10k-labels-idx1-ubyte.gz'

# 데이터 가져오기
x_train = load_mnist_images(train_images_path)
t_train = load_mnist_labels(train_labels_path)
x_test = load_mnist_images(test_images_path)
t_test = load_mnist_labels(test_labels_path)

# 데이터 확인
print(f'Train images shape: {x_train.shape}')
print(f'Train labels shape: {t_train.shape}')
print(f'Test images shape: {x_test.shape}')
print(f'Test labels shape: {t_test.shape}')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 이미지 띄우기
index = 0  # 0번 인덱스를 선택
image = x_train[index][0, :, :]
label = t_train[index]
plt.imshow(image)
plt.show()
```

<br/>

### 사용한 함수
- Affine (계층 클래스)
- ReLu (활성화 함수 클래스)
- softmax (출력층 활성화 함)
- cross_entropy_error (교차 엔트로피 오차 / 손실 함수)
- numerical_gradient (수치 미분 및 기울)
- SoftmaxWithLoss (활성화 함수 클래)
- AdaGrad (최적화 함수)
- He_init (ReLu 가중치 초기화 함수)
- 배치 정규화 (예제 소스 활용)

### CNN 함수, 클래스
- SimpleConvNet (CNN 계층 클래스)
- Convolution (합성곱 계층 클래스)
- Pooling (풀링 클래스)
- im2col (n차원 -> 2차원)
- col2im (2차원 -> n차)
