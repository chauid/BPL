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
	/// ����ġ �ʱ�ȭ ���
	/// </summary>
	enum class WeightInitMethod
	{
		ZeroInitialize, // 0 �ʱ�ȭ
		NormalDistribution, // ���� ����
		UniformDistribution, // �յ� ����
		XavierUniformDistribution,
		XavierNormalDistribution, // sigmoid, tanh�� ����
		He // LeRU�� ����
	};

	/// <summary>
	/// Ȱ��ȭ �Լ� ����
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
	/// �ս� �Լ� ����
	/// </summary>
	enum class LossFunction
	{
		MSE, // Mean Square Error | ȸ��(Regression) ����
		BinaryCrossEntropyLoss, // Ȯ�� ���� ���� ���� ��� | ���� �з�(Binary Classification) ����
		CategoricalCrossEntropyLoss, // ���� Ŭ������ ���� Ȯ�� ���� ���� ���̸� ��� | ���� Ŭ���� �з�(Multi-class Classification) ����
		SparseCrossEntropyLoss, // ��Ƽ Ŭ���� �з�
		HingeLoss // SVM(Support Vector Machine)���� ��� | ���� �з�(Binary Classification) ����
	};

	/// <summary>
	/// ����ȭ ��� ����
	/// </summary>
	enum class Optimizer
	{
		// ����ϰ���(Gradient Descent)
		// W(t + 1) = W(t) - �� * ��Loss(W(t))
		GD,
		// Ȯ���� ����ϰ���(Stochastic Gradient Descent)
		SGD,
		// �����
		// v(t + 1) = �� * v(t) - �� * ��Loss(W(t))
		// W(t + 1) = W(t) + v(t + 1)
		Momentum,
		// Adaptive Gradient Descent
		// G(t + 1) = G(t) + ��Loss(W(t))��
		// W(t + 1) = W(t) - �� * ��Loss(W(t)) / (��(G(t)) + ��)
		AdaGrad,
		// 0�� �������� �ʴ� AdaGrad
		// G(t + 1) = �� * G(t) + (1 - ��) * ��Loss(W(t))��
		// W(t + 1) = W(t) - �� * ��Loss(W(t)) / (��(G(t)) + ��)
		RMSProp,
		// Adaptive Momentum Estimation(RMSProp + Momentum)
		// m(t + 1) = ��1 * m(t) + (1 - ��1) * ��Loss(W(t))
		// v(t + 1) = ��2 * v(t) + (1 - ��2) * ��Loss(W(t))��
		// m_hat = m(t + 1) / (1 - ��1)
		// v_hat = v(t + 1) / (1 - ��2)
		// W(t + 1) = W(t) - �� * m_hat / (��(v_hat) + ��)
		Adam
	};

	/// <summary>
	/// ������ ����
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
	/// ���� �н� ���� �Ű�����
	/// </summary>
	struct early_stopping_parameters
	{
		size_t patience = 0; // �������� �н��ϴ� �ִ� �н� Ƚ�� | �� ��ġ��ŭ �н� ����� �������� ������ �н� ����
		data_type min_loss = 0.0000001; // 1e-07: ������ �����ϴ� �ּ� ������ | ���� �н����� �������� �� ��ġ��ŭ �������� ������ patience += 1
		int start_from_epoch = 0; // ���� �н� ���Ḧ ���� �ʰ� ������ �н��� �н� Ƚ��
		bool restore_best_weights = false; // patience��ŭ ����ġ�� �����ϰ� �ּ��� ������� ���� | �� ��ġ�� �����ϸ� ����ؾ��� ����ġ ����� �þ(�޸𸮰� ���� �ʿ���)
	};

	/// <summary>
	/// �� ������ �Ķ����
	/// </summary>
	struct hyper_parameters
	{
		data_type learning_rate = 0.001; // �н��� | default: 0.001
		WeightInitMethod weight_init = WeightInitMethod::UniformDistribution; // ����ġ �ʱ�ȭ | default: UniformDistribution
		LossFunction loss_function = LossFunction::MSE; // �ս� �Լ� | default: MSE
		Optimizer optimizer = Optimizer::GD; // ����ȭ ��� | default: GD
		data_type beta1 = 0.9; // Momentum: rho(��), RMSProps: learning rate decay, Adam: Beta1
		data_type beta2 = 0.999; // Adam: Beta2
	};

	/// <summary>
	/// �н� ��� �ɼ�
	/// </summary>
	struct learning_verbose_parameters
	{
		//bool verbose = false; // �ܼ��� �н� ���൵�� ǥ��: �������� ������� ����: �н��ӵ��� ������ ��ġ�� ���� | ��½ð��� step�� �ð� ��궧���� �н��� ������ | error_verbose == true�̸� verbose = true
		bool error_verbose = false; // �н� ���� ��� ����: �� �н����� ������ ����� ���� �ս� �Լ� ��� | ������ + ������ => ������ + ���� ��� + ������ | ��½ð��� step�� ���� ��궧���� �н��� ���� ������(�� ���� ����)
		bool write_file = false; // ������ ���� ��� ����: �� �н����� ������ ����� ���� �ս� �Լ� ��� | error_verbose�� ���� ���� ��
		const char* write_file_name = "learning.txt"; // 1Epoch���� �������� �����ϴ� ���ϸ�
	};

	/// <summary>
	/// �ӽ� ���� ��
	/// </summary>
	class Model
	{
	private:
		string version_info = "0.1";
		size_t number_of_input_node = 0; // �Է��� ��� ��
		size_t number_of_hidden_layer = 0; // ������ ����
		vector<size_t> number_of_nodes; // �� �������� ��� ��
		size_t max_nodes_length = 0; // ������ �� ���� ū ��� ��

		size_t dataset_size = 0; // ��ü �����ͼ� ũ��
		data_type learning_rate = 0.001; // �н��� | default: 0.001
		data_type beta1 = 0.9; // Momentum: rho(��), RMSProps: learning rate decay, Adam: Beta1
		data_type beta2 = 0.999; // Adam: Beta2
		const data_type epsilon = 0.0000001; // 1e-07
		WeightInitMethod weight_init = WeightInitMethod::UniformDistribution; // ����ġ �ʱ�ȭ | default: UniformDistribution
		LossFunction loss_function = LossFunction::MSE; // �ս� �Լ� | default: MSE
		Optimizer optimizer = Optimizer::GD; // ����ȭ ��� | default: GD

		vector<vector<data_type>> target; // ������(����) | 2���� = �����ͼ� �ε���, 1���� = ���� �ε���
		vector<vector<data_type>> input_layer; // �Է��� | 2���� = �����ͼ� �ε���, 1���� = �Է� �ε���(�Է� ��� ����)
		vector<vector<data_type>> test_layer; // �Է��� | 2���� = �׽�Ʈ�� �ε���, 1���� = �Է� �ε���(�Է� ��� ����)
		data_type** out_hidden_layer = nullptr; // �������� ��°� ���: 2���� = ���̾� �ε���, 1���� = �ش� ���̾��� �� ��� ��°�: out[��³�� �ε���: m, ��ġ �ε���: n]: o11, o21, o31, ... om1, ... o12, o22, 32, ... omn
		data_type** out_hidden_layer_no_batch = nullptr; // batch_size ���� ��� ���
		data_type** delta_hidden_layer = nullptr; // �������� ��Ÿ�� ���: 1���� = �������� ���̾� �ε���, 2���� = �� ����� ��Ÿ��
		data_type** bias_matrix = nullptr; // �������� ���Ⱚ ���: 2���� = ���̾� �ε���, 1���� = �ش� ���̾��� ��� ��� ���Ⱚ: ��ġ �����ŭ ���� �ݺ�(���� ����: �ش� ������ ��� ��)
		data_type** bias_matrix_no_batch = nullptr; // batch_size ���� ���� ���
		data_type** weight_matrix = nullptr; // ����ġ ���(������): 2���� = ���̾� �ε���(������ ���̾� = �����), 1���� = W[�������: m, ��߳��: n]: w11, w12, w13, ... w1n, w21, w22, 23, ... w2n, ... wmn
		data_type** momentum_matrix = nullptr; // Momentum v ��� �Ǵ� Adam m ���: 2���� = �������� ���̾� �ε���, 1���� = ���� ����� �ε���, ��� ����� �ε��� | ����ġ ��İ� ũ�� ����
		data_type** squared_gradient_matrix = nullptr; // Adagrad, RMSProp G ��� �Ǵ� Adam v ���: 2���� = �������� ���̾� �ε���, 1���� = ���� ����� �ε���, ��� ����� �ε��� | ����ġ ��İ� ũ�� ����

		vector<ActiveFunction> active_method_matrix; // �� ���̾��� Ȱ��ȭ �Լ�: 1���� =  �������� ���̾� �ε���
		data_type** best_weight_matrix = nullptr; // �ּ��� ����ġ ��� ����

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
			InputDataError, // �Է� ������ ����
			LoadModelOrPrepareError, // ���� �غ���� ����
			TestLengthMatchError, // ���� �Է� ���¿� �׽�Ʈ ������ �Է� ���°� ��ġ���� ����
			OuputLengthMatchError, // ���� ��� ���¿� �н� �������� ������ ���°� ��ġ���� ����
			FileOpenError, // ������ ã�� �� ���ų� ���� �� ����
			NoCotentError, // ���Ͽ� �����Ͱ� ����
			FileContentError, // �߸��� ���� ����
			ModelModifiedError, // �ҷ��� ���̳� �غ�� ���� ������ ������ �� ����
			NoLayerSetError, // ���� �غ��Ϸ��� �ּ� 1�� �̻��� �������� �־�� ��
			InvalidParamsError, // �߸��� �Ķ���� �Է�
			FunctionError, // �Լ� ����
			InSufficientMemory, // �޸� �Ҵ� ����
			DataError, // �־��� ������ ó�� ����
			CudaError, // CUDA Error
			CuBLASError // CuBLAS Error
		};

		/// <summary>
		/// ���� ó�� | return���� ���α׷� ����
		/// </summary>
		/// <param name="error_source">������ �߻��� �޼���</param>
		/// <param name="error_type">���� ����</param>
		/// <param name="param1">�޽��� �Ű����� 1</param>
		void errorHandling(string error_source, ErrorType error_type, string param1 = "");

		void checkCUDA(cudaError_t status);
		void checkCUBLAS(cublasStatus_t status);

		/// <summary>
		/// �׽�Ʈ ������ �б� | ���ϸ�: test.dat
		/// <para>** test.dat **</para>
		/// <para>0: length of input(int) | �Է°� ����</para>
		/// <para>1... : input1 input2 ...</para>
		/// <para>N: EOF</para>
		/// </summary>
		/// <param name="file_name">���ϸ�</param>
		void readTestData(const char* file_name);

		/// <summary>
		/// ������: ������ �Է� �����Ϳ� ���� out ��� | batch_size ����: ��³�� ������ 1���� ����
		/// </summary>
		/// <param name="input_data"></param>
		void forwardPropagationFromVector(vector<data_type> input_data);

		/// <summary>
		/// ������: net, out ���
		/// </summary>
		/// <param name="dataset_index">�н� �����ͼ� �ε���</param>
		/// <param name="batch_size">��ġ ������</param>
		/// <param name="loss">������ ��� | default: nullptr(��� �� ��)</param>
		/// <param name="accuracy">��Ȯ�� ��� | default: nullptr(��� �� ��)</param>
		void forwardPropagation(size_t dataset_index, size_t batch_size, data_type* loss = nullptr, data_type* accuracy = nullptr);

		/// <summary>
		/// ������: delta ���, ����ġ ����
		/// </summary>
		/// <param name="dataset_index">�н� �����ͼ� �ε���</param>
		/// <param name="batch_size">��ġ ������</param>
		/// <param name="t">���� iteration</param>
		void backPropagation(size_t dataset_index, size_t batch_size, int t);
	public:
		bool debug_mode = false; // ��� ���� ��� ���
		bool verbose_time = false; // �ð� ���

		Model() { if (sizeof(data_type) == 8) this->is_double_type = true; }
		~Model() { clearModel(); }

		void version();

		/// <summary>
		/// �� ���� �ʱ�ȭ: �޸� ��ȯ | �Է°��� �ʱ�ȭ���� ����.
		/// </summary>
		void clearModel();

		/// <summary>
		/// ���� �� �ҷ����� | �ڼ��� ������ README#Configuration�� �����ϼ���.
		/// </summary>
		/// <param name="file_name">�ҷ��� �� ���ϸ�</param>
		void loadModel(string file_name = "model.dat");

		/// <summary>
		/// ������ �߰�
		/// </summary>
		/// <param name="node_count">�ش� �������� ���(����) ��</param>
		/// <param name="active_function">�ش� �������� Ȱ��ȭ �Լ�</param>
		void addDenseLayer(int node_count, ActiveFunction active_function);

		/// <summary>
		///  1���� �����ͷ� ��ȯ: x[1][1], x[1][2],..., x[1][n] x[2][1],... -> x1, x2, x3.. x[n * m]
		/// </summary>
		void addFlattenLayer();

		void addDropoutLayer(float dropout_rate);

		/// <summary>
		/// �н� ������ �б� | ���ϸ�: input.dat
		/// <para>** input.dat **</para>
		/// <para>0: length of input(int) | �Է°� ����, length of target(int) | ��°�(����) ����</para>
		/// <para>1... : input1 input2 ... target1 target2 ...</para>
		/// <para>N: EOF</para>
		/// </summary>
		/// <param name="file_name">���ϸ�</param>
		/// <param name="print_input_data">�Էµ� ������ ��� ����</param>
		void readInputData(const char* file_name = "input.dat", bool print_input_data = false);

		/// <summary>
		/// �� ������ ����
		/// </summary>
		/// <param name="file_name">���ϸ�</param>
		void saveModel(const char* file_name = "model.dat");

		/// <summary>
		/// �� �н� �غ� = model.compile
		/// <para>�޸� �Ҵ�</para>
		/// <para>����ġ �ʱ�ȭ</para>
		/// <para>������ �Ķ���� �Է�</para>
		/// </summary>
		/// <param name="hyper_parameters">������ �Ķ���� �Է�</param>
		void prepare(hyper_parameters params = hyper_parameters());

		/// <summary>
		/// �� ���� ���
		/// </summary>
		void printModel();

		/// <summary>
		/// �� �н�
		/// </summary>
		/// <param name="epoch">�н� Ƚ��</param>
		/// <param name="batch_size">�����ͼ��� ���� ũ��(���� ����) | default: �����ͼ��� ũ��� ����(1)</param>
		/// <param name="early_stopping">���� �н� ����</param>
		/// <param name="verbose_option">�н� ��� �ɼ� | �н��ð� = max(�н��ð�, ��½ð�) | ���� �����͸� ����� ���� verbose �ɼ� ���� �� ��.</param>
		void learning(size_t epoch, size_t batch_size = 1, early_stopping_parameters early_stopping = early_stopping_parameters(), learning_verbose_parameters verbose_option = learning_verbose_parameters());

		/// <summary>
		/// �� ���� | ���� �׽�Ʈ ���̽�
		/// </summary>
		/// <param name="testdata">�׽�Ʈ ������</param>
		/// <returns>�Է°��� �� ������</returns>
		vector<data_type> predict(vector<data_type> testdata);

		/// <summary>
		/// �׽�Ʈ ������ ���Ͽ��� �� ���� | ���� �׽�Ʈ ���̽�
		/// </summary>
		/// <param name="test_file">�׽�Ʈ ������ ���ϸ�</param>
		/// <returns>�׽�Ʈ �������� �� ������ ����Ʈ</returns>
		vector<vector<data_type>> predictFromFile(const char* test_file = "test.dat");

		/// <summary>
		/// ������ ������ ���Ϸ� ��� | ���� �׽�Ʈ ���̽�
		/// </summary>
		/// <param name="testdata">�׽�Ʈ ������</param>
		/// <param name="delimiter">���� ����</param>
		/// <param name="output_file">������ ��� ���ϸ�</param>
		/// <param name="write_with_testdata">��� ���Ͽ� �׽�Ʈ ������ ���� ����</param>
		void predictToFile(vector<data_type> testdata, const char* delimiter = " ", const char* output_file = "ouput.txt", bool write_with_testdata = false);

		/// <summary>
		/// �׽�Ʈ ������ ���Ͽ��� ������ ������ ���Ϸ� ��� | ���� �׽�Ʈ ���̽�
		/// </summary>
		/// <param name="test_file">�׽�Ʈ ������ ���ϸ�</param>
		/// <param name="delimiter">���� ����</param>
		/// <param name="output_file">������ ��� ���ϸ�</param>
		/// <param name="write_with_testdata">��� ���Ͽ� �׽�Ʈ ������ ���� ����</param>
		void predictToFileFromFile(const char* test_file = "test.dat", const char* delimiter = " ", const char* output_file = "ouput.txt", bool write_with_testdata = false);
	};
}

#endif // !__MODEL__