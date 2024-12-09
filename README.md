## Usage
|사용 메서드|반환 자료형|설명|선행 조건|
|---|:---:|---|---|
|clearModel|void|모델 정보 초기화|없음|
|loadModel|void|기존 모델 불러오기|없음|
|addLayer|void|은닉층 추가|없음|
|addDropoutLayer|void||없음|
|readInputData|void|학습 데이터 읽기|없음|
|printInputData|void|입력된 학습 데이터 출력|readInputData|
|saveModel|void|모델 데이터 저장|readInputData|
|prepare|void|모델 학습 준비|readInputData, (loadModel or addLayer)|
|printModel|void|모델 구조 출력|(loadModel or prepare)|
|learning|void|모델 학습|(loadModel or prepare)|
|calcLoss|double|전체 데이터셋의 오차율 평균 계산|(loadModel or prepare)|
|predict|vector<double>|값 예측|(loadModel or prepare)|
|predictFromFile|vector<vector<double>>|테스트 데이터 파일에서 값 예측|(loadModel or prepare)|
|predictToFile|void|예측값 데이터 파일로 출력|(loadModel or prepare)|
|predictToFileFromFile|void|테스트 데이터 파일에서 예측값 데이터 파일로 출력|(loadModel or prepare)|

> [!WARNING]
> prepare 후 loadModel를 실행하면 현재 모델 구조와 가중치 정보를 불러오는 기존의 모델로 덮어씁니다. (모델 구조 및 가중치값 손실)  
> 기존에 학습한 모델의 정보를 저장(saveModel) 후 loadModel을 사용하십시오.  
> **손실된 값은 복구할 수 없습니다.**  

## Configuration
`input.data`
> [입력 길이] [출력 길이]  
> 입력값1 입력값2 ... 출력값1 출력값2

### e.g.  
입력값: [[0, 0], [0, 1], [1, 0], [1, 1]]  
출력값: [0, 1, 1, 0]  
_(리스트의 개수는 학습데이터의 개수를 의미)_  
```
** input.data **
2 1
0 0 0
0 1 1
1 0 1
1 1 0
```

<hr />

`test.data`
> [입력 길이]
> 입력값1 입력값2 ...

`input.data`에서 출력값만 제외한 형태
```
** test.data **
2
0 0
0 1
1 0
1 1
```

<hr />

`model.data`
> [입력 길이] [은닉층 개수]  
> [각 은닉층의 노드 수: 1레이어 노드 수, 2레이어 노드수, ...]  
> [각 은닉층의 활성화 함수: 1레이어 활성화 함수, 2레이어 활성화 함수, ...]  
> [weight_init] [optimizer] [loss_function] [learning_rate]  
> [가중치11, 가중치12, ..., 가중치1n] [bias_1] < `1번째 레이어의 가중치 행렬`  
> [가중치21, 가중치22, ..., 가중치2n] [bias_2]  
> [가중치... ] [bias... ]  
> [가중치m1, 가중치22, ..., 가중치mn] [bias_m]  
> 
> [속도11, 속도12, ..., 속도1n] < `1번째 레이어의 속도 행렬`  
> [속도21, 속도22, ..., 속도2n]  
> [속도... ]  
> [속도m1, 속도22, ..., 속도mn]  
> 
> [가중치11, 가중치12, ..., 가중치1n] [bias_1] < `2번째 레이어의 가중치 행렬`  
> [가중치21, 가중치22, ..., 가중치2n] [bias_2]  
> [ 가중치... ] [ bias... ]  
> [가중치m1, 가중치22, ..., 가중치mn] [bias_m]  
> 
> . . .  

_','는 편의상 README에서만 표현.(파일에는 ','를 넣지 않음)_  
k번째 은닉층에서 출발노드(k-1번 은닉층) 인덱스를 n, 도착노드(k번째 은닉층) 인덱스를 m으로 표시  
k번째 은닉층의 편향값 개수 = k번째 은닉층의 노드 수  
모델을 로드 후 이어서 학습하기 위해 속도 행렬(학습의 진행도)를 저장함.  

|Active Function(활성화 함수 목록)|Value|
|---|---|
|Sigmoid|0|
|HyperbolicTangent|1|
|Softmax|2|
|LeRU|3|
|leakyReLU|4|
|ELU|5|

|Weight Initialization(가중치 초기화 방법)|Value|
|---|---|
|UniformDistribution|0|
|NormalDistribution|1|
|XavierGlorot|2|
|He|3|

|Optimizer(최적화 기법 목록)|Value|
|---|---|
|GD|0|
|SGD|1|
|Momentum|2|
|RMSProps|3|
|Adam|4|

|Loss Function(손실 함수 목록)|Value|
|---|---|
|MSE|0|
|Binary CrossEntropy|1|
|HingeLoss|2|
|CrossEntropyLoss|3|
|Sparse CrossEntropyLoss|4|

### e.g.  
|Layer|(Input, Output) Shape|Param|
|---|---|---|
|Input Layer|(0, 2)|None| 
|Hidden Layer1|(2, 2)|6|
|Hidden Layer2(Ouput Layer)|(2, 1)|3|

위와 같은 모델 구조에 1번 레이어의 활성화 함수를 Sigmoid, 2번 레이어의 활성화 함수를 ReLU로 설정  
weight_init: UniformDistribution,  
optimizer: Momentum,  
loss_function: MSE,  
learning_rate: 0.0015,   
```
** model.data **
2 2
2 1
0 3
0 2 0 0.0015
5.179993 5.134700 0.830690
0.851149 0.850280 0.750170

0.000012 0.000011
0.000083 0.000157

8.280300 -9.082571 0.224180

-0.000134 -0.000220
```