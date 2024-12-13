#pragma once
#ifndef __MODEL__
#define __MODEL__

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <thread>
#include <cublas_v2.h>

#include "Functions.h"

using namespace std;
using namespace bpl;

using data_type = double;

namespace bpl
{
	/// <summary>
	/// 가중치 초기화 방법
	/// </summary>
	enum class WeightInitMethod
	{
		ZeroInitialize, // 0 초기화
		NormalDistribution, // 정규 분포
		UniformDistribution, // 균등 분포
		XavierUniformDistribution,
		XavierNormalDistribution, // sigmoid, tanh에 적합
		He // LeRU에 적합
	};

	/// <summary>
	/// 활성화 함수 선택
	/// </summary>
	enum class ActiveFunction
	{
		Sigmoid, // 1 / (1 + exp(-x))
		HyperbolicTangent, // (2 / 1 + exp(-2x)) - 1 = 2 * sigmoid(2 * x) - 1
		Softmax, // Index of each x in list: n | exp(x[n] - x.max) / sum(exp(x[n] - x.max))
		ReLU, // x > 0 ? x : 0
		leakyReLU, // x > 0 ? x : 0.1x
		ELU, // x > 0 ? x : exp(x) - 1
		Swish // x * (1 / (1 + exp(-x))) = x * sigmoid(x)
	};

	/// <summary>
	/// 손실 함수 선택
	/// </summary>
	enum class LossFunction
	{
		MSE, // Mean Square Error | 회귀(Regression) 문제
		BinaryCrossEntropyLoss, // 확률 분포 간의 차이 계산 | 이진 분류(Binary Classification) 문제
		CategoricalCrossEntropyLoss, // 실제 클래스와 예측 확률 분포 간의 차이를 계산 | 다중 클래스 분류(Multi-class Classification) 문제
		SparseCrossEntropyLoss, // 멀티 클래스 분류
		HingeLoss // SVM(Support Vector Machine)에서 사용 | 이진 분류(Binary Classification) 문제
	};

	/// <summary>
	/// 최적화 기법 선택
	/// </summary>
	enum class Optimizer
	{
		// 경사하강법(Gradient Descent)
		// W(t + 1) = W(t) - η * ∇Loss(W(t))
		GD,
		// 확률적 경사하강법(Stochastic Gradient Descent)
		SGD,
		// 모멘텀
		// v(t + 1) = ρ * v(t) - η * ∇Loss(W(t))
		// W(t + 1) = W(t) + v(t + 1)
		Momentum,
		// Adaptive Gradient Descent
		// G(t + 1) = G(t) + ∇Loss(W(t))²
		// W(t + 1) = W(t) - η * ∇Loss(W(t)) / (√(G(t)) + ε)
		AdaGrad,
		// 0에 수렴하지 않는 AdaGrad
		// G(t + 1) = β * G(t) + (1 - β) * ∇Loss(W(t))²
		// W(t + 1) = W(t) - η * ∇Loss(W(t)) / (√(G(t)) + ε)
		RMSProp,
		// Adaptive Momentum Estimation(RMSProp + Momentum)
		// m(t + 1) = β1 * m(t) + (1 - β1) * ∇Loss(W(t))
		// v(t + 1) = β2 * v(t) + (1 - β2) * ∇Loss(W(t))²
		// m_hat = m(t + 1) / (1 - β1)
		// v_hat = v(t + 1) / (1 - β2)
		// W(t + 1) = W(t) - η * m_hat / (√(v_hat) + ε)
		Adam
	};

	/// <summary>
	/// 은닉층 유형
	/// </summary>
	enum class LayerType
	{
		Dense,
		Convolution,
		SimpleRNN,
		LSTM,
		Dropout,
		Flatten
	};

	/// <summary>
	/// 조기 학습 종료 매개변수
	/// </summary>
	struct early_stopping_parameters
	{
		size_t patience = 0; // 개선없이 학습하는 최대 학습 횟수 | 이 수치만큼 학습 결과가 개선되지 않으면 학습 종료
		data_type min_loss = 0.0000001; // 1e-07: 개선을 인정하는 최소 오차값 | 다음 학습에서 오차값이 이 수치만큼 개선되지 않으면 patience += 1
		int start_from_epoch = 0; // 조기 학습 종료를 하지 않고 무조건 학습할 학습 횟수
		bool restore_best_weights = false; // patience만큼 가중치를 저장하고 최선의 결과값을 도출 | 이 수치가 증가하면 백업해야할 가중치 행렬이 늘어남(메모리가 많이 필요함)
	};

	/// <summary>
	/// 모델 하이퍼 파라미터
	/// </summary>
	struct hyper_parameters
	{
		data_type learning_rate = 0.001; // 학습률 | default: 0.001
		WeightInitMethod weight_init = WeightInitMethod::UniformDistribution; // 가중치 초기화 | default: UniformDistribution
		LossFunction loss_function = LossFunction::MSE; // 손실 함수 | default: MSE
		Optimizer optimizer = Optimizer::GD; // 최적화 기법 | default: GD
		data_type beta1 = 0.9; // Momentum: rho(ρ), RMSProps: learning rate decay, Adam: Beta1
		data_type beta2 = 0.999; // Adam: Beta2
	};

	/// <summary>
	/// 학습 출력 옵션
	/// </summary>
	struct learning_verbose_parameters
	{
		//bool verbose = false; // 단순히 학습 진행도만 표시: 오차율을 계산하지 않음: 학습속도에 영향을 미치지 않음 | 출력시간과 step당 시간 계산때문에 학습이 느려짐 | error_verbose == true이면 verbose = true
		bool error_verbose = false; // 학습 과정 출력 여부: 매 학습마다 오차율 계산을 위해 손실 함수 계산 | 순전파 + 역전파 => 순전파 + 오차 계산 + 역전파 | 출력시간과 step당 오차 계산때문에 학습이 조금 느려짐(별 차이 없음)
		bool write_file = false; // 오차율 파일 출력 여부: 매 학습마다 오차율 계산을 위해 손실 함수 계산 | error_verbose에 비해 빠른 편
		const char* write_file_name = "learning.txt"; // 1Epoch마다 오차값을 저장하는 파일명
	};

	/// <summary>
	/// 머신 러닝 모델
	/// </summary>
	class Model
	{
	private:
		string version_info = "0.1";
		size_t number_of_input_node = 0; // 입력층 노드 수
		size_t number_of_hidden_layer = 0; // 은닉층 개수
		vector<size_t> number_of_nodes; // 각 은닉층의 노드 수
		size_t max_nodes_length = 0; // 은닉층 중 가장 큰 노드 수

		size_t dataset_size = 0; // 전체 데이터셋 크기
		data_type learning_rate = 0.001; // 학습률 | default: 0.001
		data_type beta1 = 0.9; // Momentum: rho(ρ), RMSProps: learning rate decay, Adam: Beta1
		data_type beta2 = 0.999; // Adam: Beta2
		const data_type epsilon = 0.0000001; // 1e-07
		WeightInitMethod weight_init = WeightInitMethod::UniformDistribution; // 가중치 초기화 | default: UniformDistribution
		LossFunction loss_function = LossFunction::MSE; // 손실 함수 | default: MSE
		Optimizer optimizer = Optimizer::GD; // 최적화 기법 | default: GD

		vector<vector<data_type>> target; // 실제값(정답) | 2차원 = 데이터셋 인덱스, 1차원 = 정답 인덱스
		vector<vector<data_type>> input_layer; // 입력층 | 2차원 = 데이터셋 인덱스, 1차원 = 입력 인덱스(입력 노드 길이)
		vector<vector<data_type>> test_layer; // 입력층 | 2차원 = 테스트셋 인덱스, 1차원 = 입력 인덱스(입력 노드 길이)
		data_type** out_hidden_layer = nullptr; // 은닉층의 출력값 행렬: 2차원 = 레이어 인덱스, 1차원 = 해당 레이어의 각 노드 출력값: out[출력노드 인덱스: m, 배치 인덱스: n]: o11, o21, o31, ... om1, ... o12, o22, 32, ... omn
		data_type** out_hidden_layer_no_batch = nullptr; // batch_size 없는 출력 행렬
		data_type** delta_hidden_layer = nullptr; // 은닉층의 델타값 행렬: 1차원 = 은닉층의 레이어 인덱스, 2차원 = 각 노드의 델타값
		data_type** bias_matrix = nullptr; // 은닉층의 편향값 행렬: 2차원 = 레이어 인덱스, 1차원 = 해당 레이어의 출력 노드 편향값: 배치 사이즈만큼 값을 반복(기존 차원: 해당 은닉층 노드 수)
		data_type** bias_matrix_no_batch = nullptr; // batch_size 없는 편향 행렬
		data_type** weight_matrix = nullptr; // 가중치 행렬(은닉층): 2차원 = 레이어 인덱스(마지막 레이어 = 출력층), 1차원 = W[도착노드: m, 출발노드: n]: w11, w12, w13, ... w1n, w21, w22, 23, ... w2n, ... wmn
		data_type** momentum_matrix = nullptr; // Momentum v 행렬 또는 Adam m 행렬: 2차원 = 은닉층의 레이어 인덱스, 1차원 = 도착 노드의 인덱스, 출발 노드의 인덱스 | 가중치 행렬과 크기 동일
		data_type** squared_gradient_matrix = nullptr; // Adagrad, RMSProp G 행렬 또는 Adam v 행렬: 2차원 = 은닉층의 레이어 인덱스, 1차원 = 도착 노드의 인덱스, 출발 노드의 인덱스 | 가중치 행렬과 크기 동일

		vector<ActiveFunction> active_method_matrix; // 각 레이어의 활성화 함수: 1차원 =  은닉층의 레이어 인덱스
		data_type** best_weight_matrix = nullptr; // 최선의 가중치 행렬 저장

		cublasHandle_t handle = nullptr;
		data_type* dev_A = nullptr; // device A: Forward,weight matrix(trans) || Back_delta,weight matrix || Back_weight,weight matrix
		data_type* dev_B = nullptr; // device B: Forward,out matrix || Back_delta,out delta matrix || Back_weight,weight matrix
		data_type* dev_C = nullptr; // device C: Forward,bias matrix || Back_delta,out delta matrix || Back_weight,weight matrix
		data_type* dev_D = nullptr; // device D: Back_weight,weight matrix
		data_type* result_C = nullptr; // host_C || host_y | Forward: net | Back: delta, gradient
		data_type* host_memory = nullptr; // temp host memory: weight length
		bool is_prepared = false;
		bool is_loaded = false;
		bool is_double_type = false;

		enum class ErrorType
		{
			InputDataError, // 입력 데이터 없음
			LoadModelOrPrepareError, // 모델이 준비되지 않음
			TestLengthMatchError, // 모델의 입력 형태와 테스트 데이터 입력 형태가 일치하지 않음
			OuputLengthMatchError, // 모델의 출력 형태와 학습 데이터의 실제값 형태가 일치하지 않음
			FileOpenError, // 파일을 찾을 수 없거나 읽을 수 없음
			NoCotentError, // 파일에 데이터가 없음
			FileContentError, // 잘못된 파일 내용
			ModelModifiedError, // 불러온 모델이나 준비된 모델은 구조를 변경할 수 없음
			NoLayerSetError, // 모델을 준비하려면 최소 1개 이상의 은닉층이 있어야 함
			InvalidParamsError, // 잘못된 파라미터 입력
			FunctionError, // 함수 에러
			InSufficientMemory, // 메모리 할당 에러
			DataError, // 주어진 데이터 처리 에러
			CudaError, // CUDA Error
			CuBLASError // CuBLAS Error
		};

		/// <summary>
		/// 에러 처리 | return없이 프로그램 종료
		/// </summary>
		/// <param name="error_source">에러가 발생한 메서드</param>
		/// <param name="error_type">에러 유형</param>
		/// <param name="param1">메시지 매개변수 1</param>
		void errorHandling(string error_source, ErrorType error_type, string param1 = "");

		void checkCUDA(cudaError_t status);
		void checkCUBLAS(cublasStatus_t status);

		/// <summary>
		/// 테스트 데이터 읽기 | 파일명: test.dat
		/// <para>** test.dat **</para>
		/// <para>0: length of input(int) | 입력값 길이</para>
		/// <para>1... : input1 input2 ...</para>
		/// <para>N: EOF</para>
		/// </summary>
		/// <param name="file_name">파일명</param>
		void readTestData(const char* file_name);

		/// <summary>
		/// 순전파: 임의의 입력 데이터에 대한 out 계산 | batch_size 없음: 출력노드 차원은 1차원 고정
		/// </summary>
		/// <param name="input_data"></param>
		void forwardPropagationFromVector(vector<data_type> input_data);

		/// <summary>
		/// 순전파: net, out 계산
		/// </summary>
		/// <param name="dataset_index">학습 데이터셋 인덱스</param>
		/// <param name="batch_size">배치 사이즈</param>
		/// <param name="loss">오차값 계산 | default: nullptr(계산 안 함)</param>
		/// <param name="accuracy">정확도 계산 | default: nullptr(계산 안 함)</param>
		void forwardPropagation(size_t dataset_index, size_t batch_size, data_type* loss = nullptr, data_type* accuracy = nullptr);

		/// <summary>
		/// 역전파: delta 계산, 가중치 갱신
		/// </summary>
		/// <param name="dataset_index">학습 데이터셋 인덱스</param>
		/// <param name="batch_size">배치 사이즈</param>
		/// <param name="t">현재 iteration</param>
		void backPropagation(size_t dataset_index, size_t batch_size, int t);
	public:
		bool debug_mode = false; // 계산 과정 모두 출력
		bool verbose_time = false; // 시간 출력

		Model() { if (sizeof(data_type) == 8) this->is_double_type = true; }
		~Model() { clearModel(); }

		void version();

		/// <summary>
		/// 모델 정보 초기화: 메모리 반환 | 입력값은 초기화하지 않음.
		/// </summary>
		void clearModel();

		/// <summary>
		/// 기존 모델 불러오기 | 자세한 내용은 README#Configuration를 참고하세요.
		/// </summary>
		/// <param name="file_name">불러올 모델 파일명</param>
		void loadModel(string file_name = "model.dat");

		/// <summary>
		/// 은닉층 추가
		/// </summary>
		/// <param name="node_count">해당 은닉층의 노드(뉴런) 수</param>
		/// <param name="active_function">해당 은닉층의 활성화 함수</param>
		void addDenseLayer(int node_count, ActiveFunction active_function);

		/// <summary>
		///  1차원 데이터로 변환: x[1][1], x[1][2],..., x[1][n] x[2][1],... -> x1, x2, x3.. x[n * m]
		/// </summary>
		void addFlattenLayer();

		void addDropoutLayer(float dropout_rate);

		/// <summary>
		/// 학습 데이터 읽기 | 파일명: input.dat
		/// <para>** input.dat **</para>
		/// <para>0: length of input(int) | 입력값 길이, length of target(int) | 출력값(정답) 길이</para>
		/// <para>1... : input1 input2 ... target1 target2 ...</para>
		/// <para>N: EOF</para>
		/// </summary>
		/// <param name="file_name">파일명</param>
		/// <param name="print_input_data">입력된 데이터 출력 여부</param>
		void readInputData(const char* file_name = "input.dat", bool print_input_data = false);

		/// <summary>
		/// 모델 데이터 저장
		/// </summary>
		/// <param name="file_name">파일명</param>
		void saveModel(const char* file_name = "model.dat");

		/// <summary>
		/// 모델 학습 준비 = model.compile
		/// <para>메모리 할당</para>
		/// <para>가중치 초기화</para>
		/// <para>하이퍼 파라미터 입력</para>
		/// </summary>
		/// <param name="hyper_parameters">하이퍼 파라미터 입력</param>
		void prepare(hyper_parameters params = hyper_parameters());

		/// <summary>
		/// 모델 구조 출력
		/// </summary>
		void printModel();

		/// <summary>
		/// 모델 학습
		/// </summary>
		/// <param name="epoch">학습 횟수</param>
		/// <param name="batch_size">데이터셋을 묶는 크기(병렬 연산) | default: 데이터셋의 크기와 동일(1)</param>
		/// <param name="early_stopping">조기 학습 종료</param>
		/// <param name="verbose_option">학습 출력 옵션 | 학습시간 = max(학습시간, 출력시간) | 적은 데이터를 사용할 때는 verbose 옵션 권장 안 함.</param>
		void learning(size_t epoch, size_t batch_size = 1, early_stopping_parameters early_stopping = early_stopping_parameters(), learning_verbose_parameters verbose_option = learning_verbose_parameters());

		/// <summary>
		/// 값 예측 | 단일 테스트 케이스
		/// </summary>
		/// <param name="testdata">테스트 데이터</param>
		/// <returns>입력값의 모델 예측값</returns>
		vector<data_type> predict(vector<data_type> testdata);

		/// <summary>
		/// 테스트 데이터 파일에서 값 예측 | 다중 테스트 케이스
		/// </summary>
		/// <param name="test_file">테스트 데이터 파일명</param>
		/// <returns>테스트 데이터의 모델 예측값 리스트</returns>
		vector<vector<data_type>> predictFromFile(const char* test_file = "test.dat");

		/// <summary>
		/// 예측값 데이터 파일로 출력 | 단일 테스트 케이스
		/// </summary>
		/// <param name="testdata">테스트 데이터</param>
		/// <param name="delimiter">구분 문자</param>
		/// <param name="output_file">예측값 출력 파일명</param>
		/// <param name="write_with_testdata">출력 파일에 테스트 데이터 포함 여부</param>
		void predictToFile(vector<data_type> testdata, const char* delimiter = " ", const char* output_file = "ouput.txt", bool write_with_testdata = false);

		/// <summary>
		/// 테스트 데이터 파일에서 예측값 데이터 파일로 출력 | 다중 테스트 케이스
		/// </summary>
		/// <param name="test_file">테스트 데이터 파일명</param>
		/// <param name="delimiter">구분 문자</param>
		/// <param name="output_file">예측값 출력 파일명</param>
		/// <param name="write_with_testdata">출력 파일에 테스트 데이터 포함 여부</param>
		void predictToFileFromFile(const char* test_file = "test.dat", const char* delimiter = " ", const char* output_file = "ouput.txt", bool write_with_testdata = false);
	};
}

#endif // !__MODEL__