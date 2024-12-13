#include "Model.h"

void bpl::Model::errorHandling(string error_source, ErrorType error_type, string param1)
{
	printf("[%s]에서 오류 발생: ", error_source.c_str());
	switch (error_type)
	{
	case ErrorType::InputDataError:
		printf("입력 데이터 없음.\n선행 조건: readInputData()\n");
		break;
	case ErrorType::LoadModelOrPrepareError:
		printf("모델이 준비되지 않음.\n선행 조건: loadModel() 또는 prepare()\n");
		break;
	case ErrorType::TestLengthMatchError:
		printf("모델의 입력 형태와 테스트 데이터 입력 형태가 일치하지 않음.\n선행 조건: 모델 입력값 길이와 테스트 데이터의 입력 길이가 동일해야 함.\n");
		break;
	case ErrorType::OuputLengthMatchError:
		printf("모델의 출력 형태와 학습 데이터의 실제값 형태가 일치하지 않음.\n선행 조건: 모델 출력값 길이와 학습 데이터의 출력 길이가 동일해야 함.\nHint: 마지막 addLayer()를 확인하세요.\n");
		break;
	case ErrorType::FileOpenError:
		printf("\"%s\" 파일을 찾을 수 없거나 읽을 수 없음.\n", param1.c_str());
		break;
	case ErrorType::NoCotentError:
		printf("\"%s\" 파일에 데이터가 없음.\n", param1.c_str());
		break;
	case ErrorType::FileContentError:
		printf("\"%s\" 파일에서 잘못된 내용.\n", param1.c_str());
		break;
	case ErrorType::ModelModifiedError:
		printf("불러온 모델이나 준비된 모델은 구조를 변경할 수 없음. 새로운 모델 객체를 생성하거나 모델 정보 초기화 후 시도하세요.\nHint: clearModel()");
		break;
	case ErrorType::NoLayerSetError:
		printf("모델을 준비하려면 최소 1개 이상의 은닉층(출력층)이 있어야 함.\n");
		break;
	case ErrorType::InvalidParamsError:
		printf("잘못된 파라미터 입력. 오류가 발생한 파라미터: %s\n", param1.c_str());
		break;
	case ErrorType::FunctionError:
		printf("오류 발생 함수: %s\n", param1.c_str());
		break;
	case ErrorType::InSufficientMemory:
		printf("메모리 할당 오류: %s\n", param1.c_str());
		break;
	case ErrorType::DataError:
		printf("데이터 오류: %s\n", param1.c_str());
		break;
	case ErrorType::CudaError:
		printf("Error Number: %s\n", param1.c_str());
		break;
	case ErrorType::CuBLASError:
		printf("Error Number: %s\n", param1.c_str());
		break;
	default:
		printf("알 수 없는 오류가 발생\n"); // 절대 실행될 일 없음.
		break;
	}

	exit(1);
}

void bpl::Model::checkCUDA(cudaError_t status)
{
	if (status != cudaSuccess) errorHandling("checkCUDA", ErrorType::CudaError, to_string(status));
}

void bpl::Model::checkCUBLAS(cublasStatus_t status)
{
	if (status != CUBLAS_STATUS_SUCCESS) errorHandling("checkCUBLAS", ErrorType::CuBLASError, to_string(status));
}

void bpl::Model::readTestData(const char* file_name)
{
	FILE* pFile = nullptr;
	size_t input_length = 0; // 입력값 개수(노드 수) | 한 줄에 최대 입력 수
	size_t input_layer_length = this->number_of_input_node; // 모델 입력값 개수(노드 수)
	size_t sum_of_count = 0; // 전체 파일 입력 개수 | 파일 전체 값 % 입력값 == 0 ? pass : error

	data_type buffer; // 입력 버퍼

	if (fopen_s(&pFile, file_name, "r") != 0) { errorHandling("readTestData", ErrorType::FileOpenError, file_name); return; }
	fscanf_s(pFile, "%zu", &input_length); // 테스트 데이터 구조 읽기

	if (input_length != input_layer_length) { fclose(pFile);  errorHandling("readTestData", ErrorType::TestLengthMatchError, file_name); }

	vector<data_type> input_line(input_length); // 학습데이터 입력 레이어 1개
	int cols = 0; // 한 줄에 입력받는 값 인덱스
	while (fscanf_s(pFile, "%lf", &buffer) != EOF)
	{
		sum_of_count++;
		if (cols < input_length) input_line[cols] = buffer;

		if (cols == input_length - 1)
		{
			this->test_layer.push_back(input_line);
			cols = 0;
		}
		else cols++;
	}
	fclose(pFile);

	if (sum_of_count == 0) errorHandling("readTestData", ErrorType::NoCotentError, file_name);
	if (sum_of_count % input_length != 0) errorHandling("readTestData", ErrorType::FileContentError, file_name);
}

void bpl::Model::forwardPropagationFromVector(vector<data_type> input_data)
{
	data_type alpha = 1.0; // C = α * A * B + β * C
	data_type beta = 1.0; // C = α * A * B + β * C
	size_t input_length = this->number_of_input_node; // 입력 차원 길이
	size_t output_length = this->number_of_nodes[0]; // 출력 차원 길이

	// CUBLAS 반환값 행렬 메모리 할당
	this->result_C = (data_type*)malloc(sizeof(data_type) * this->max_nodes_length * this->max_nodes_length);
	if (this->result_C == NULL) { errorHandling("learning", ErrorType::InSufficientMemory, "result_C"); return; }

	checkCUDA(cudaMalloc((void**)&this->dev_A, this->max_nodes_length * this->max_nodes_length * sizeof(data_type)));
	checkCUDA(cudaMalloc((void**)&this->dev_B, this->max_nodes_length * sizeof(data_type)));
	checkCUDA(cudaMalloc((void**)&this->dev_C, this->max_nodes_length * sizeof(data_type)));

	//Functions::printArray("A", output_length * input_length, this->weight_matrix[0], input_length);
	//Functions::printArray("B", input_length, &input_data[0], input_length);
	//Functions::printArray("C", output_length, this->bias_matrix_no_batch[0], output_length);

	cublasSetMatrix(input_length, output_length, sizeof(data_type), this->weight_matrix[0], input_length, this->dev_A, input_length);
	cublasSetMatrix(input_length, 1, sizeof(data_type), &input_data[0], input_length, this->dev_B, input_length);
	cublasSetMatrix(output_length, 1, sizeof(data_type), this->bias_matrix_no_batch[0], output_length, this->dev_C, output_length);

	cublasDgemm(this->handle, CUBLAS_OP_T, CUBLAS_OP_N, output_length, 1, input_length, &alpha, this->dev_A, input_length, this->dev_B, input_length, &beta, this->dev_C, output_length);
	cublasGetMatrix(output_length, 1, sizeof(data_type), this->dev_C, output_length, this->result_C, output_length);

	//Functions::printArray("result C", output_length, this->result_C, output_length);

	for (size_t p = 0; p < output_length; p++)
	{
		switch (this->active_method_matrix[0])
		{
		case ActiveFunction::Sigmoid:
			this->out_hidden_layer_no_batch[0][p] = Functions::Sigmoid(this->result_C[p]);
			break;
		case ActiveFunction::HyperbolicTangent:
			this->out_hidden_layer_no_batch[0][p] = Functions::tanh(this->result_C[p]);
			break;
		case ActiveFunction::Softmax:
			if (p == 0) Functions::Softmax(this->out_hidden_layer_no_batch[0], output_length);
			break;
		case ActiveFunction::ReLU:
			this->out_hidden_layer_no_batch[0][p] = Functions::ReLU(this->result_C[p]);
			break;
		case ActiveFunction::leakyReLU:
			this->out_hidden_layer_no_batch[0][p] = Functions::leakyReLU(this->result_C[p]);
			break;
		case ActiveFunction::ELU:
			this->out_hidden_layer_no_batch[0][p] = Functions::ELU(this->result_C[p]);
			break;
		case ActiveFunction::Swish:
			this->out_hidden_layer_no_batch[0][p] = Functions::Swish(this->result_C[p]);
			break;
		default:
			printf("알 수 없는 활성화 함수. 프로그램 종료.\n");
			exit(1);
			break;
		}
	}

	// 은닉층의 1번째부터 마지막 레이어까지 순전파
	for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		input_length = this->number_of_nodes[layer_index - 1]; // 입력 차원 길이
		output_length = this->number_of_nodes[layer_index]; // 출력 차원 길이

		checkCUBLAS(cublasSetMatrix(input_length, output_length, sizeof(data_type), this->weight_matrix[layer_index], input_length, this->dev_A, input_length));
		checkCUBLAS(cublasSetMatrix(input_length, 1, sizeof(data_type), this->out_hidden_layer_no_batch[layer_index - 1], input_length, this->dev_B, input_length));
		checkCUBLAS(cublasSetMatrix(output_length, 1, sizeof(data_type), this->bias_matrix_no_batch[layer_index], output_length, this->dev_C, output_length));

		//Functions::printArray("A", output_length * input_length, this->weight_matrix[layer_index], input_length);
		//Functions::printArray("B", input_length, this->out_hidden_layer_no_batch[layer_index - 1], input_length);
		//Functions::printArray("C", output_length, this->bias_matrix_no_batch[layer_index], output_length);
		cublasDgemm(this->handle, CUBLAS_OP_T, CUBLAS_OP_N, output_length, 1, input_length, &alpha, this->dev_A, input_length, this->dev_B, input_length, &beta, this->dev_C, output_length);
		cublasGetMatrix(output_length, 1, sizeof(data_type), this->dev_C, output_length, this->result_C, output_length);

		//Functions::printArray("result C", output_length, this->result_C, output_length);

		for (size_t p = 0; p < output_length; p++)
		{
			switch (this->active_method_matrix[layer_index])
			{
			case ActiveFunction::Sigmoid:
				this->out_hidden_layer_no_batch[layer_index][p] = Functions::Sigmoid(this->result_C[p]);
				break;
			case ActiveFunction::HyperbolicTangent:
				this->out_hidden_layer_no_batch[layer_index][p] = Functions::tanh(this->result_C[p]);
				break;
			case ActiveFunction::Softmax:
				if (p == 0) Functions::Softmax(this->out_hidden_layer_no_batch[layer_index], output_length);
				break;
			case ActiveFunction::ReLU:
				this->out_hidden_layer_no_batch[layer_index][p] = Functions::ReLU(this->result_C[p]);
				break;
			case ActiveFunction::leakyReLU:
				this->out_hidden_layer_no_batch[layer_index][p] = Functions::leakyReLU(this->result_C[p]);
				break;
			case ActiveFunction::ELU:
				this->out_hidden_layer_no_batch[layer_index][p] = Functions::ELU(this->result_C[p]);
				break;
			case ActiveFunction::Swish:
				this->out_hidden_layer_no_batch[layer_index][p] = Functions::Swish(this->result_C[p]);
				break;
			default:
				printf("알 수 없는 활성화 함수. 프로그램 종료.\n");
				exit(1);
				break;
			}
		}
	}
	checkCUDA(cudaFree(this->dev_A));
	checkCUDA(cudaFree(this->dev_B));
	checkCUDA(cudaFree(this->dev_C));

	free(this->result_C);
}

void bpl::Model::forwardPropagation(size_t dataset_index, size_t batch_size, data_type* loss, data_type* accuracy)
{
	data_type alpha = 1.0; // C = α * A * B + β * C
	data_type beta = 1.0; // C = α * A * B + β * C
	size_t input_length = this->number_of_input_node; // 입력 차원 길이
	size_t output_length = this->number_of_nodes[0]; // 출력 차원 길이

	data_type** copy_input = (data_type**)malloc(sizeof(data_type*) * batch_size);
	if (copy_input == NULL) { errorHandling("forwardPropagation", ErrorType::InSufficientMemory, "copy_input"); return; }
	data_type* host_B0 = (data_type*)malloc(sizeof(data_type) * input_length * batch_size);
	if (host_B0 == NULL) { errorHandling("forwardPropagation", ErrorType::InSufficientMemory, "host_B"); return; }
	for (size_t batch_index = 0; batch_index < batch_size; batch_index++) copy_input[batch_index] = &this->input_layer[dataset_index + batch_index][0];
	for (size_t batch_index = 0; batch_index < batch_size; batch_index++)
	{
		for (size_t p = 0; p < input_length; p++) host_B0[batch_index * input_length + p] = copy_input[batch_index][p];
	}

	//Functions::printArray("A", output_length * input_length, this->weight_matrix[0], input_length);
	//Functions::printArray("B", input_length * batch_size, host_B0, input_length);
	//Functions::printArray("C", output_length * batch_size, this->bias_matrix[0], output_length);

	cublasSetMatrix(input_length, output_length, sizeof(data_type), this->weight_matrix[0], input_length, this->dev_A, input_length);
	cublasSetMatrix(input_length, batch_size, sizeof(data_type), host_B0, input_length, this->dev_B, input_length);
	cublasSetMatrix(output_length, batch_size, sizeof(data_type), this->bias_matrix[0], output_length, this->dev_C, output_length);

	cublasDgemm(this->handle, CUBLAS_OP_T, CUBLAS_OP_N, output_length, batch_size, input_length, &alpha, this->dev_A, input_length, this->dev_B, input_length, &beta, this->dev_C, output_length);
	cublasGetMatrix(output_length, batch_size, sizeof(data_type), this->dev_C, output_length, this->result_C, output_length);

	//Functions::printArray("result C", output_length * batch_size, this->result_C, output_length);

	for (size_t batch_index = 0; batch_index < batch_size; batch_index++)
	{
		for (size_t p = 0; p < output_length * batch_size; p++)
		{
			switch (this->active_method_matrix[0])
			{
			case ActiveFunction::Sigmoid:
				this->out_hidden_layer[0][p] = Functions::Sigmoid(this->result_C[p]);
				break;
			case ActiveFunction::HyperbolicTangent:
				this->out_hidden_layer[0][p] = Functions::tanh(this->result_C[p]);
				break;
			case ActiveFunction::Softmax:
				if (p == 0) // batch 1번에 1번만 계산
				{
					data_type* single_out = (data_type*)malloc(sizeof(data_type) * output_length);
					if (single_out == NULL) { errorHandling("forwardPropagation", ErrorType::InSufficientMemory, "single_out"); return; }
					for (size_t out_index = batch_index * output_length, i = 0; out_index < (batch_index + 1) * output_length; out_index++, i++)
					{
						single_out[i] = this->result_C[out_index];
					}
					Functions::Softmax(single_out, output_length);
					for (size_t softmax_p = 0; softmax_p < output_length; softmax_p++)
					{
						this->out_hidden_layer[0][batch_index * output_length + p] = single_out[softmax_p];
					}
					free(single_out);
				}
				break;
			case ActiveFunction::ReLU:
				this->out_hidden_layer[0][p] = Functions::ReLU(this->result_C[p]);
				break;
			case ActiveFunction::leakyReLU:
				this->out_hidden_layer[0][p] = Functions::leakyReLU(this->result_C[p]);
				break;
			case ActiveFunction::ELU:
				this->out_hidden_layer[0][p] = Functions::ELU(this->result_C[p]);
				break;
			case ActiveFunction::Swish:
				this->out_hidden_layer[0][p] = Functions::Swish(this->result_C[p]);
				break;
			default:
				printf("알 수 없는 활성화 함수. 프로그램 종료.\n");
				exit(1);
				break;
			}
		}
	}
	free(copy_input);
	free(host_B0);

	// 은닉층의 1번째부터 마지막 레이어까지 순전파
	for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		input_length = this->number_of_nodes[layer_index - 1]; // 입력 차원 길이
		output_length = this->number_of_nodes[layer_index]; // 출력 차원 길이

		cublasSetMatrix(input_length, output_length, sizeof(data_type), this->weight_matrix[layer_index], input_length, this->dev_A, input_length);
		cublasSetMatrix(input_length, batch_size, sizeof(data_type), this->out_hidden_layer[layer_index - 1], input_length, this->dev_B, input_length);
		cublasSetMatrix(output_length, batch_size, sizeof(data_type), this->bias_matrix[layer_index], output_length, this->dev_C, output_length);

		//Functions::printArray("A", output_length * input_length, this->weight_matrix[layer_index], input_length);
		//Functions::printArray("B", input_length * batch_size, this->out_hidden_layer[layer_index - 1], input_length);
		//Functions::printArray("C", output_length * batch_size, this->bias_matrix[layer_index], output_length);

		cublasDgemm(this->handle, CUBLAS_OP_T, CUBLAS_OP_N, output_length, batch_size, input_length, &alpha, this->dev_A, input_length, this->dev_B, input_length, &beta, this->dev_C, output_length);
		cublasGetMatrix(output_length, batch_size, sizeof(data_type), this->dev_C, output_length, this->result_C, output_length);

		//Functions::printArray("result C", output_length * batch_size, this->result_C, output_length);

		for (size_t batch_index = 0; batch_index < batch_size; batch_index++)
		{
			for (size_t p = 0; p < output_length; p++)
			{
				switch (this->active_method_matrix[layer_index])
				{
				case ActiveFunction::Sigmoid:
					this->out_hidden_layer[layer_index][batch_index * output_length + p] = Functions::Sigmoid(this->result_C[batch_index * output_length + p]);
					break;
				case ActiveFunction::HyperbolicTangent:
					this->out_hidden_layer[layer_index][batch_index * output_length + p] = Functions::tanh(this->result_C[batch_index * output_length + p]);
					break;
				case ActiveFunction::Softmax:
					if (p == 0) // batch 1번에 1번만 계산
					{
						data_type* single_out = (data_type*)malloc(sizeof(data_type) * output_length);
						if (single_out == NULL) { errorHandling("forwardPropagation", ErrorType::InSufficientMemory, "single_out"); return; }
						for (size_t out_index = batch_index * output_length, i = 0; out_index < (batch_index + 1) * output_length; out_index++, i++)
						{
							single_out[i] = this->result_C[out_index];
						}
						Functions::Softmax(single_out, output_length);
						for (size_t softmax_p = 0; softmax_p < output_length; softmax_p++)
						{
							this->out_hidden_layer[layer_index][batch_index * output_length + p] = single_out[softmax_p];
						}
						free(single_out);
					}
					break;
				case ActiveFunction::ReLU:
					this->out_hidden_layer[layer_index][batch_index * output_length + p] = Functions::ReLU(this->result_C[batch_index * output_length + p]);
					break;
				case ActiveFunction::leakyReLU:
					this->out_hidden_layer[layer_index][batch_index * output_length + p] = Functions::leakyReLU(this->result_C[batch_index * output_length + p]);
					break;
				case ActiveFunction::ELU:
					this->out_hidden_layer[layer_index][batch_index * output_length + p] = Functions::ELU(this->result_C[batch_index * output_length + p]);
					break;
				case ActiveFunction::Swish:
					this->out_hidden_layer[layer_index][batch_index * output_length + p] = Functions::Swish(this->result_C[batch_index * output_length + p]);
					break;
				default:
					printf("알 수 없는 활성화 함수. 프로그램 종료.\n");
					exit(1);
					break;
				}
				if (this->debug_mode) printf("active(net[0][%zu](%lf)) + bias(%lf)) = %lf\n", p, this->result_C[batch_index * output_length + p], bias_matrix[layer_index][p * batch_size + batch_index], out_hidden_layer[layer_index][p * batch_size + batch_index] - bias_matrix[layer_index][p * batch_size + batch_index]);
			}
		}
		if (this->debug_mode) printf("\n");
	}

	if (loss != nullptr)
	{
		data_type* output_layer = new data_type[this->number_of_nodes.back()];
		switch (loss_function)
		{
		case LossFunction::MSE:
			*loss = Functions::MSE(&this->target[dataset_index][0], this->out_hidden_layer[this->number_of_hidden_layer - 1], this->number_of_nodes.back());
			for (size_t batch_index = 1; batch_index < batch_size; batch_index++)
			{
				for (size_t p = 0; p < this->number_of_nodes.back(); p++) output_layer[p] = this->out_hidden_layer[this->number_of_hidden_layer - 1][batch_index * output_length + p];
				*loss += Functions::MSE(&this->target[dataset_index + batch_index][0], output_layer, this->number_of_nodes.back());
			}
			*loss /= batch_size;
			break;
		case LossFunction::BinaryCrossEntropyLoss:
			*loss = Functions::BinaryCrossEntropy(&this->target[dataset_index][0], this->out_hidden_layer[this->number_of_hidden_layer - 1], this->number_of_nodes.back());
			for (size_t batch_index = 1; batch_index < batch_size; batch_index++)
			{
				for (size_t p = 0; p < this->number_of_nodes.back(); p++) output_layer[p] = this->out_hidden_layer[this->number_of_hidden_layer - 1][batch_index * output_length + p];
				*loss += Functions::BinaryCrossEntropy(&this->target[dataset_index + batch_index][0], output_layer, this->number_of_nodes.back());
			}
			*loss /= batch_size;
			break;
		case LossFunction::HingeLoss:
			break;
		case LossFunction::CategoricalCrossEntropyLoss:
			*loss = Functions::CategoricalCrossEntropy(&this->target[dataset_index][0], this->out_hidden_layer[this->number_of_hidden_layer - 1], this->number_of_nodes.back());
			for (size_t batch_index = 1; batch_index < batch_size; batch_index++)
			{
				for (size_t p = 0; p < this->number_of_nodes.back(); p++) output_layer[p] = this->out_hidden_layer[this->number_of_hidden_layer - 1][batch_index * output_length + p];
				*loss += Functions::CategoricalCrossEntropy(&this->target[dataset_index + batch_index][0], output_layer, this->number_of_nodes.back());
			}
			*loss /= batch_size;
			break;
		case LossFunction::SparseCrossEntropyLoss:
			break;
		default:
			break;
		}
		delete[] output_layer;
	}
}

void bpl::Model::backPropagation(size_t dataset_index, size_t batch_size, int t)
{
	data_type active_diff = 0.0; // 활성화 함수 미분

	size_t target_length = this->number_of_nodes.back();
	data_type out_p = 0.0; // 해당 레이어의 p번째 노드의 출력값(예측값)
	data_type target_p = 0.0; // 마지막 레이어의 p번째 노드의 실제값(정답)

	data_type out_j = 0.0; // k번째 레이어의 j번째 노드의 출력값(예측값)
	const data_type alpha = 1.0; // C = α * A * B + β * C
	const data_type beta = 0.0; // C = α * A * B + β * C
	size_t input_length = this->number_of_nodes[this->number_of_hidden_layer - 1]; // 입력 차원 길이
	size_t output_length = this->number_of_nodes.back(); // 출력 차원 길이

	chrono::system_clock::time_point start1 = chrono::system_clock::now();
	/* delta 계산 */
	for (size_t batch_index = 0; batch_index < batch_size; batch_index++)
	{
		for (size_t p = 0; p < target_length; p++) // p: 마지막 레이어의 노드 인덱스
		{
			out_p = this->out_hidden_layer[this->number_of_hidden_layer - 1][batch_index * target_length + p];
			target_p = this->target[dataset_index + batch_index][p];

			switch (this->active_method_matrix.back())
			{
			case ActiveFunction::Sigmoid:
				active_diff = out_p * (1 - out_p); // dx/x sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
				if (active_diff < this->epsilon) active_diff = this->epsilon;
				if (active_diff > 1 - this->epsilon) active_diff = 1 - this->epsilon;
				if (debug_mode) cout << "active diff[0] = " << out_p << "*" << (1 - out_p) << '\n';
				break;
			case ActiveFunction::HyperbolicTangent:
				active_diff = (1 - out_p) * (1 + out_p);  // dx/x tanh(x) = (1 - tanh(x)) * (1 + tanh(x))
				break;
			case ActiveFunction::Softmax:
				if (input_length == output_length)
				{

				}
				printf("Softmax 아직 구현 안 됨\n");
				break;
			case ActiveFunction::ReLU:
				active_diff = out_p > 0 ? 1 : 0; // dx/x ReLU(x) = x > 0 ? 1 : 0
				break;
			case ActiveFunction::leakyReLU:
				active_diff = out_p > 0 ? 1 : 0.1; // dx/x leakyReLU(x) = x > 0 ? 1 : 0.1
				break;
			case ActiveFunction::ELU:
				active_diff = out_p > 0 ? 1 : out_p + 1; // dx/x ELU(x) = x > 0 ? 1 : exp(x)
				break;
			case ActiveFunction::Swish:
				// Swish(x) = y = x * sigmoid(x) = x * (1 + e^-x)
				// y = x / (1 + e^-x)
				// y <-> x
				// y / (1 + e^-y) = x
				// y = x(1 + e^-y)
				// y = x + xe^-y
				// y = W(x / e^x) + x
				// y = W(xe^(-x)) + x
				active_diff = out_p; // dx/x Swish(x) = sigmoid(x) * Swish(x) * (1 - sigmoid(x))
				break;
			default:
				printf("알 수 없는 활성화 함수. 프로그램 종료.\n");
				exit(1);
				break;
			}
			//printf("\nout: %.9lf\n", out_p);

			switch (this->loss_function)
			{
			case LossFunction::MSE:
				// δ[last_index][p] = ∂Error[p]/∂out[p] = -(2 / pn) * (실제값 - 예측값) * (예측값 * (1 - 예측값)) | -(2 / pn) * (실제값 - 예측값) = MSE 도함수
				// 손실함수 미분 * 활성화함수 미분 * (-1)
				this->delta_hidden_layer[this->number_of_hidden_layer - 1][batch_index * target_length + p] = -(2.0 / target_length) * (target_p - out_p) * active_diff;
				if (debug_mode) cout << "delta hidden layer= " << -(2.0 / target_length) << "*" << (target_p - out_p) << "*" << active_diff << '\n';
				break;
			case LossFunction::BinaryCrossEntropyLoss:
				// target == 1, (target / output)
				// target == 0, -(1 - target) / (1 - output)
				// ∂Error[p]/∂out[p] = -(1 / pn) * ()
				if (out_p < this->epsilon) out_p = this->epsilon;
				if (out_p > (1 - this->epsilon)) out_p = 1 - this->epsilon;
				this->delta_hidden_layer[this->number_of_hidden_layer - 1][batch_index * target_length + p] = -(target_p / out_p) + (1 - target_p) / (1 - out_p);
				//if (target_p == 1.0) this->delta_hidden_layer[this->number_of_hidden_layer - 1][batch_index * target_length + p] = -(target_p / out_p);
				//else if (target_p == 0.0) this->delta_hidden_layer[this->number_of_hidden_layer - 1][batch_index * target_length + p] = (1 - target_p) / (1 - out_p);
				//else { errorHandling("backPropagation", ErrorType::DataError, "실제값은 One-Hot Vector이어야 함"); return; }
				break;
			case LossFunction::HingeLoss:
				printf("HingeLoss 아직 구현 안 됨\n");
				break;
			case LossFunction::CategoricalCrossEntropyLoss:
				// CrossEntropy + Softmax => ∂Error[p]/∂out[p] = output - target
				if (this->active_method_matrix.back() == ActiveFunction::Softmax)
				{
					this->delta_hidden_layer[this->number_of_hidden_layer - 1][batch_index * target_length + p] = out_p - target_p;
				}
				else this->delta_hidden_layer[this->number_of_hidden_layer - 1][batch_index * target_length + p] = -(target_p / out_p);
				break;
			case LossFunction::SparseCrossEntropyLoss:
				printf("SparseCrossEntropyLoss 아직 구현 안 됨\n");
				break;
			default:
				printf("알 수 없는 손실 함수. 프로그램 종료.\n");
				exit(1);
				break;
			}
			if (debug_mode) printf("output layer delta batch[0] = %.9lf\n", this->delta_hidden_layer[this->number_of_hidden_layer - 1][batch_index * target_length + p]);
		}
	}

	// [kn - 1]은 출력층의 델타값, [kn - 2]부터 은닉층의 델타값 계산 (0 <= layer_index < (kn: 은닉층의 개수))
	for (int layer_index = this->number_of_hidden_layer - 2; layer_index >= 0; layer_index--)
	{
		input_length = this->number_of_nodes[layer_index];
		output_length = this->number_of_nodes[layer_index + 1];

		cublasSetMatrix(input_length, output_length, sizeof(data_type), this->weight_matrix[layer_index + 1], input_length, this->dev_A, input_length);
		cublasSetMatrix(output_length, batch_size, sizeof(data_type), this->delta_hidden_layer[layer_index + 1], output_length, this->dev_B, output_length);
		cublasSetMatrix(input_length, batch_size, sizeof(data_type), this->result_C, input_length, this->dev_C, input_length);

		if (debug_mode) Functions::printArray("가중치 행렬", input_length * this->number_of_input_node, this->weight_matrix[layer_index + 1], input_length);
		if (debug_mode) Functions::printArray("델타 행렬", output_length * batch_size, this->delta_hidden_layer[layer_index + 1], output_length);

		cublasDgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_N, input_length, batch_size, output_length, &alpha, this->dev_A, input_length, this->dev_B, output_length, &beta, this->dev_C, input_length);
		cublasGetMatrix(input_length, batch_size, sizeof(data_type), this->dev_C, input_length, this->result_C, input_length);

		for (size_t batch_index = 0; batch_index < batch_size; batch_index++)
		{
			for (size_t j = 0; j < input_length; j++) // j: 출발 노드의 인덱스
			{
				out_j = this->out_hidden_layer[layer_index][batch_index * input_length + j];

				switch (this->active_method_matrix[layer_index])
				{
				case ActiveFunction::Sigmoid:
					active_diff = out_j * (1 - out_j); // dx/x sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
					if (active_diff < this->epsilon) active_diff = this->epsilon;
					if (active_diff > 1 - this->epsilon) active_diff = 1 - this->epsilon;
					if (debug_mode) cout << "active diff[" << j << "]=" << out_j << "*" << (1 - out_j) << '\n';
					break;
				case ActiveFunction::HyperbolicTangent:
					active_diff = (1 - out_j) * (1 + out_j); // dx/x tanh(x) = (1 - tanh(x)) * (1 + tanh(x))
					break;
				case ActiveFunction::Softmax:
					printf("Softmax 함수는 출력층에서만 사용할 수 있음. 프로그램 종료.\n");
					exit(1);
					break;
				case ActiveFunction::ReLU:
					active_diff = out_j > 0 ? 1 : 0; // dx/x ReLU(x) = x > 0 ? 1 : 0
					if (this->debug_mode) printf(" * (%lf > 0 ? 1 : 0) (%lf)\n", out_hidden_layer[layer_index][j], active_diff);
					break;
				case ActiveFunction::leakyReLU:
					active_diff = out_j > 0 ? 1 : 0.1; // dx/x leakyReLU(x) = x > 0 ? 1 : 0.1
					if (this->debug_mode) printf(" * (%lf > 0 ? 1 : 0.1) (%lf)\n", out_hidden_layer[layer_index][j], active_diff);
					break;
				case ActiveFunction::ELU:
					active_diff = out_j > 0 ? 1 : out_j + 1; // dx/x ELU(x) = x > 0 ? 1 : exp(x)
					if (this->debug_mode) printf(" * (%lf > 0 ? 1 : 0.1) (%lf)\n", out_hidden_layer[layer_index][j], active_diff);
					break;
				default:
					printf("알 수 없는 활성화 함수. 프로그램 종료.\n");
					exit(1);
					break;
				}
				this->delta_hidden_layer[layer_index][batch_index * input_length + j] = result_C[batch_index * input_length + j] * active_diff;
				if (debug_mode) printf("delta batch[%zu] = %.15lf, result_C[%zu]: %.15lf\n", batch_index, this->delta_hidden_layer[layer_index][batch_index * input_length + j], j, result_C[batch_index * input_length + j]);
			}
		}
	}

	/* 가중치 갱신 */
	size_t weight_length = this->number_of_nodes[0] * this->number_of_input_node;
	data_type scalar_alpha = 0.0; // cublas<t>axpy alpha
	data_type gradient_scalar = 0.0; // ∇Loss(W(t)) = ∂Error/∂W(t) = δ(t) * ∂Hnet(t)/∂W(t) = δ[k][p] * Hout[k - 1][j] | ∂Hnet/∂W = Hout[k - 1]
	data_type divide_batch = 1.0 / batch_size; // 1 / batch_size
	data_type m_hat = 0.0;
	data_type v_hat = 0.0;

	input_length = this->number_of_input_node;
	output_length = this->number_of_nodes[0];
	weight_length = output_length * input_length;

	data_type** copy_input = (data_type**)malloc(sizeof(data_type*) * batch_size);
	if (copy_input == NULL) { errorHandling("backPropagation", ErrorType::InSufficientMemory, "copy_input"); return; }
	data_type* host_A = (data_type*)malloc(sizeof(data_type) * input_length * batch_size);
	if (host_A == NULL) { errorHandling("backPropagation", ErrorType::InSufficientMemory, "host_A"); return; }
	for (size_t batch_index = 0; batch_index < batch_size; batch_index++) copy_input[batch_index] = &this->input_layer[dataset_index + batch_index][0];
	for (size_t batch_index = 0; batch_index < batch_size; batch_index++)
	{
		for (size_t p = 0; p < input_length; p++) host_A[batch_index * input_length + p] = copy_input[batch_index][p];
	}

	cublasSetMatrix(input_length, batch_size, sizeof(data_type), host_A, input_length, this->dev_A, input_length);
	cublasSetMatrix(output_length, batch_size, sizeof(data_type), this->delta_hidden_layer[0], output_length, this->dev_B, output_length);
	cublasSetMatrix(input_length, output_length, sizeof(data_type), this->result_C, input_length, this->dev_C, input_length);

	cublasDgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_T, input_length, output_length, batch_size, &alpha, this->dev_A, input_length, this->dev_B, output_length, &beta, this->dev_C, input_length);
	cublasDscal(this->handle, weight_length, &divide_batch, this->dev_C, 1);

	if (debug_mode) Functions::printArray("기존 가중치", this->number_of_nodes[0] * this->number_of_input_node, this->weight_matrix[0], this->number_of_input_node);
	switch (this->optimizer)
	{
	case Optimizer::GD:
		// W(t + 1) = W(t) - η * ∇Loss(W(t))
		scalar_alpha = (-1) * this->learning_rate; // scalar_alpha = -1 * η
		cublasSetVector(weight_length, sizeof(data_type), this->weight_matrix[0], 1, this->dev_A, 1); // dev_A <- W(t)

		cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_A, 1); // dev_A = -1 * η * ∇Loss(W(t)) + W(t)
		cublasGetVector(weight_length, sizeof(data_type), this->dev_A, 1, this->weight_matrix[0], 1); // W(t + 1) <- dev_A
		break;
	case Optimizer::Momentum:
		// v(t + 1) = ρ * v(t) - η * ∇Loss(W(t))
		// W(t + 1) = W(t) + v(t + 1)
		scalar_alpha = (-1) * this->learning_rate; // scalar_alpha = -1 * η
		cublasSetVector(weight_length, sizeof(data_type), this->weight_matrix[0], 1, this->dev_A, 1); // dev_A <- W(t)
		cublasSetVector(weight_length, sizeof(data_type), this->momentum_matrix[0], 1, this->dev_B, 1); // dev_B <- v(t)

		cublasDscal(this->handle, weight_length, &this->beta1, this->dev_B, 1); // dev_B = ρ * v(t)
		cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_B, 1); // dev_B = (-1 * η) * ∇Loss(W(t)) + dev_B
		cublasDaxpy(this->handle, weight_length, &alpha, this->dev_B, 1, this->dev_A, 1); // dev_A = dev_B + W(t)
		cublasGetVector(weight_length, sizeof(data_type), this->dev_A, 1, this->weight_matrix[0], 1); // W(t + 1) <- dev_A
		cublasGetVector(weight_length, sizeof(data_type), this->dev_B, 1, this->momentum_matrix[0], 1); // v(t + 1) <- dev_B
		break;
	case Optimizer::AdaGrad:
		// G(t + 1) = G(t) + ∇Loss(W(t))²
		// W(t + 1) = W(t) - η * ∇Loss(W(t)) / (√(G(t + 1)) + ε)
		scalar_alpha = (-1) * this->learning_rate; // scalar_alpha = -1 * η
		cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->result_C, 1); // result_C <- ∇Loss(W(t))
		for (size_t p = 0; p < weight_length; p++) this->squared_gradient_matrix[0][p] += this->result_C[p] * this->result_C[p]; // G(t + 1) = G(t) + ∇Loss(W(t))²
		for (size_t p = 0; p < weight_length; p++) this->result_C[p] = sqrt(this->squared_gradient_matrix[0][p]) + this->epsilon; // result_C = √(G(t + 1)) + ε

		cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->host_memory, 1); // host_memory <- ∇Loss(W(t))
		for (size_t p = 0; p < weight_length; p++) this->result_C[p] = this->host_memory[p] / this->result_C[p]; // result_C = ∇Loss(W(t)) / result_C

		cublasSetVector(weight_length, sizeof(data_type), this->weight_matrix[0], 1, this->dev_A, 1); // dev_A <- W(t)
		cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_C, 1); // dev_C <- result_C

		cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_A, 1); // dev_A = (-1 * η) * dev_C + dev_A
		cublasGetVector(weight_length, sizeof(data_type), this->dev_A, 1, this->weight_matrix[0], 1); // W(t + 1) <- dev_A
		break;
	case Optimizer::RMSProp:
		// G(t + 1) = β * G(t) + (1 - β) * ∇Loss(W(t))²
		// W(t + 1) = W(t) - η * ∇Loss(W(t)) / (√(G(t + 1)) + ε)
		scalar_alpha = (-1) * this->learning_rate; // scalar_alpha = -1 * η
		cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->result_C, 1); // result_C <- ∇Loss(W(t))
		//for (size_t p = 0; p < weight_length; p++) printf("0-result_C(%lf) =  %lf * %lf * %lf\n", (1 - this->beta1) * this->result_C[p] * this->result_C[p], (1 - this->beta1), this->result_C[p], this->result_C[p]); // result_C = (1 - β) * ∇Loss(W(t))²
		for (size_t p = 0; p < weight_length; p++) this->result_C[p] = (1 - this->beta1) * this->result_C[p] * this->result_C[p]; // result_C = (1 - β) * ∇Loss(W(t))²
		cublasSetVector(weight_length, sizeof(data_type), this->squared_gradient_matrix[0], 1, this->dev_D, 1); // dev_D <- G(t)
		//Functions::printArray("0-squared_gradient_matrix", weight_length, this->squared_gradient_matrix[0], output_length);
		cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_B, 1); // dev_B <- result_C
		cublasDaxpy(this->handle, weight_length, &this->beta1, this->dev_D, 1, this->dev_B, 1); // dev_B = β * G(t) + dev_B

		cublasGetVector(weight_length, sizeof(data_type), this->dev_B, 1, this->squared_gradient_matrix[0], 1); // G(t + 1) <- dev_B
		//for (size_t p = 0; p < weight_length; p++) printf("0-√(G(t + 1)) + ε = √(%lf) + %lf\n", this->squared_gradient_matrix[0][p], this->epsilon);
		for (size_t p = 0; p < weight_length; p++) this->result_C[p] = sqrt(this->squared_gradient_matrix[0][p]) + this->epsilon; // result_C = √(G(t + 1)) + ε

		cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->host_memory, 1); // host_memory <- ∇Loss(W(t))
		for (size_t p = 0; p < weight_length; p++) this->result_C[p] = this->host_memory[p] / this->result_C[p]; // result_C = ∇Loss(W(t)) / result_C

		cublasSetVector(weight_length, sizeof(data_type), this->weight_matrix[0], 1, this->dev_A, 1); // dev_A <- W(t)
		cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_C, 1); // dev_C <- result_C

		cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_A, 1); // dev_A = (-1 * η) * dev_C + dev_A
		cublasGetVector(weight_length, sizeof(data_type), this->dev_A, 1, this->weight_matrix[0], 1); // W(t + 1) <- dev_A
		//Functions::printArray("weight_matrix", weight_length, this->weight_matrix[0], input_length);
		break;
	case Optimizer::Adam:
		// m(t + 1) = β1 * m(t) + (1 - β1) * ∇Loss(W(t))
		// v(t + 1) = β2 * v(t) + (1 - β2) * ∇Loss(W(t))²
		// m_hat = m(t + 1) / (1 - β1^t)
		// v_hat = v(t + 1) / (1 - β2^t)
		// W(t + 1) = W(t) - η * m_hat / (√(v_hat) + ε)
		scalar_alpha = 1.0 - this->beta1; // scalar_alpha = 1 - β1
		cublasSetVector(weight_length, sizeof(data_type), this->momentum_matrix[0], 1, this->dev_D, 1); // dev_D <- m(t)
		cublasDscal(this->handle, weight_length, &this->beta1, this->dev_D, 1); // dev_D = β1 * m(t)
		cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_D, 1); // dev_D = (1 - β1) * ∇Loss(W(t)) + dev_D

		cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->result_C, 1); // result_C <- ∇Loss(W(t))
		for (size_t p = 0; p < weight_length; p++) this->result_C[p] = (1 - this->beta2) * this->result_C[p] * this->result_C[p]; // result_C = (1 - β2) * ∇Loss(W(t))²
		cublasSetVector(weight_length, sizeof(data_type), this->squared_gradient_matrix[0], 1, this->dev_C, 1); // dev_C <- v(t)
		cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_B, 1); // dev_B <- result_C
		cublasDaxpy(this->handle, weight_length, &this->beta2, this->dev_C, 1, this->dev_B, 1); // dev_B = β2 * v(t) + dev_B

		cublasGetVector(weight_length, sizeof(data_type), this->dev_D, 1, this->momentum_matrix[0], 1); // m(t + 1) <- dev_D
		cublasGetVector(weight_length, sizeof(data_type), this->dev_B, 1, this->squared_gradient_matrix[0], 1); // v(t + 1) <- dev_B

		for (size_t p = 0; p < weight_length; p++) this->result_C[p] = 0; // result_C = {0,}

		scalar_alpha = 1.0 / (1.0 - pow(this->beta1, t)); // scalar_alpha = 1 / (1 - β1^t)
		cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_C, 1); // dev_C <- result_C
		cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_D, 1, this->dev_C, 1); // dev_C(m_hat) = 1 / (1 - β1^t) * dev_D + {0,}

		scalar_alpha = 1.0 / (1.0 - pow(this->beta2, t)); // scalar_alpha = 1 / (1 - β2^t)
		cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_D, 1); // dev_D <- result_C
		cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_B, 1, this->dev_D, 1); // dev_D(v_hat) = 1 / (1 - β2^t) * dev_B + {0,}

		cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->result_C, 1); // result_C <- dev_C(m_hat)
		cublasGetVector(weight_length, sizeof(data_type), this->dev_D, 1, this->host_memory, 1); // host_memory <- dev_D(v_hat)
		for (size_t p = 0; p < weight_length; p++) this->result_C[p] = this->result_C[p] / (sqrt(this->host_memory[p]) + this->epsilon); // result_C = m_hat / (√(v_hat) + ε)

		cublasSetVector(weight_length, sizeof(data_type), this->weight_matrix[0], 1, this->dev_D, 1); // dev_D <- W(t)
		cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_C, 1); // dev_C <- result_C

		scalar_alpha = (-1) * this->learning_rate; // scalar_alpha = -1 * η
		cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_D, 1); // dev_D = (-1 * η) * dev_C + dev_D
		cublasGetVector(weight_length, sizeof(data_type), this->dev_D, 1, this->weight_matrix[0], 1); // W(t + 1) <- dev_D
		break;
	default:
		printf("알 수 없는 최적화 기법. 프로그램 종료.\n");
		exit(1);
		break;
	}
	if (debug_mode) Functions::printArray("갱신 후 가중치", this->number_of_nodes[0] * this->number_of_input_node, this->weight_matrix[0], this->number_of_input_node);
	free(copy_input);
	free(host_A);

	for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		input_length = this->number_of_nodes[layer_index - 1];
		output_length = this->number_of_nodes[layer_index];
		weight_length = output_length * input_length;

		cublasSetMatrix(input_length, batch_size, sizeof(data_type), this->out_hidden_layer[layer_index - 1], input_length, this->dev_A, input_length);
		cublasSetMatrix(output_length, batch_size, sizeof(data_type), this->delta_hidden_layer[layer_index], output_length, this->dev_B, output_length);
		cublasSetMatrix(input_length, output_length, sizeof(data_type), this->result_C, input_length, this->dev_C, input_length);

		cublasDgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_T, input_length, output_length, batch_size, &alpha, this->dev_A, input_length, this->dev_B, output_length, &beta, this->dev_C, input_length);
		cublasDscal(this->handle, weight_length, &divide_batch, this->dev_C, 1);

		if (debug_mode) Functions::printArray("기존 가중치[i]", this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1], this->weight_matrix[layer_index], this->number_of_nodes[layer_index - 1]);
		switch (this->optimizer)
		{
		case Optimizer::GD:
			// W(t + 1) = W(t) - η * ∇Loss(W(t))
			scalar_alpha = (-1) * this->learning_rate; // scalar_alpha = -1 * η
			cublasSetVector(weight_length, sizeof(data_type), this->weight_matrix[layer_index], 1, this->dev_A, 1); // dev_A <- W(t)

			cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_A, 1); // dev_A = -1 * η * ∇Loss(W(t)) + W(t)
			cublasGetVector(weight_length, sizeof(data_type), this->dev_A, 1, this->weight_matrix[layer_index], 1); // W(t + 1) <- dev_A
			break;
		case Optimizer::Momentum:
			// v(t + 1) = ρ * v(t) - η * ∇Loss(W(t))
			// W(t + 1) = W(t) + v(t + 1)
			scalar_alpha = (-1) * this->learning_rate; // scalar_alpha = -1 * η
			cublasSetVector(weight_length, sizeof(data_type), this->weight_matrix[layer_index], 1, this->dev_A, 1); // dev_A <- W(t)
			cublasSetVector(weight_length, sizeof(data_type), this->momentum_matrix[layer_index], 1, this->dev_B, 1); // dev_B <- v(t)

			cublasDscal(this->handle, weight_length, &this->beta1, this->dev_B, 1); // dev_B = ρ * v(t)
			cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_B, 1); // dev_B = (-1 * η) * ∇Loss(W(t)) + dev_B
			cublasDaxpy(this->handle, weight_length, &alpha, this->dev_B, 1, this->dev_A, 1); // dev_A = dev_B + W(t)
			cublasGetVector(weight_length, sizeof(data_type), this->dev_A, 1, this->weight_matrix[layer_index], 1); // W(t + 1) <- dev_A
			cublasGetVector(weight_length, sizeof(data_type), this->dev_B, 1, this->momentum_matrix[layer_index], 1); // v(t + 1) <- dev_B
			break;
		case Optimizer::AdaGrad:
			// G(t + 1) = G(t) + ∇Loss(W(t))²
			// W(t + 1) = W(t) - η * ∇Loss(W(t)) / (√(G(t + 1)) + ε)
			scalar_alpha = (-1) * this->learning_rate; // scalar_alpha = -1 * η
			cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->result_C, 1); // result_C <- ∇Loss(W(t))
			for (size_t p = 0; p < weight_length; p++) this->squared_gradient_matrix[layer_index][p] += this->result_C[p] * this->result_C[p]; // G(t + 1) = G(t) + ∇Loss(W(t))²
			for (size_t p = 0; p < weight_length; p++) this->result_C[p] = sqrt(this->squared_gradient_matrix[layer_index][p]) + this->epsilon; // result_C = √(G(t + 1)) + ε

			cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->host_memory, 1); // host_memory <- ∇Loss(W(t))
			for (size_t p = 0; p < weight_length; p++) this->result_C[p] = this->host_memory[p] / this->result_C[p]; // result_C = ∇Loss(W(t)) / result_C

			cublasSetVector(weight_length, sizeof(data_type), this->weight_matrix[layer_index], 1, this->dev_A, 1); // dev_A <- W(t)
			cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_C, 1); // dev_C <- result_C

			cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_A, 1); // dev_A = (-1 * η) * dev_C + dev_A
			cublasGetVector(weight_length, sizeof(data_type), this->dev_A, 1, this->weight_matrix[layer_index], 1); // W(t + 1) <- dev_A
			break;
		case Optimizer::RMSProp:
			// G(t + 1) = β * G(t) + (1 - β) * ∇Loss(W(t))²
			// W(t + 1) = W(t) - η * ∇Loss(W(t)) / (√(G(t + 1)) + ε)
			scalar_alpha = (-1) * this->learning_rate; // scalar_alpha = -1 * η
			cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->result_C, 1); // result_C <- ∇Loss(W(t))
			for (size_t p = 0; p < weight_length; p++) this->result_C[p] = (1 - this->beta1) * this->result_C[p] * this->result_C[p]; // result_C = (1 - β) * ∇Loss(W(t))²
			//for (size_t p = 0; p < weight_length; p++) printf("i-%lf * %lf * %lf\n", (1 - this->beta1), this->result_C[p], this->result_C[p]); // result_C = (1 - β) * ∇Loss(W(t))²
			cublasSetVector(weight_length, sizeof(data_type), this->squared_gradient_matrix[layer_index], 1, this->dev_D, 1); // dev_D <- G(t)
			cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_B, 1); // dev_B <- result_C
			cublasDaxpy(this->handle, weight_length, &this->beta1, this->dev_D, 1, this->dev_B, 1); // dev_B = β * G(t) + dev_B

			cublasGetVector(weight_length, sizeof(data_type), this->dev_B, 1, this->squared_gradient_matrix[layer_index], 1); // G(t + 1) <- dev_B
			for (size_t p = 0; p < weight_length; p++) this->result_C[p] = sqrt(this->squared_gradient_matrix[layer_index][p]) + this->epsilon; // result_C = √(G(t + 1)) + ε

			cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->host_memory, 1); // host_memory <- ∇Loss(W(t))
			for (size_t p = 0; p < weight_length; p++) this->result_C[p] = this->host_memory[p] / this->result_C[p]; // result_C = ∇Loss(W(t)) / result_C

			cublasSetVector(weight_length, sizeof(data_type), this->weight_matrix[layer_index], 1, this->dev_A, 1); // dev_A <- W(t)
			cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_C, 1); // dev_C <- result_C

			cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_A, 1); // dev_A = (-1 * η) * dev_C + dev_A
			cublasGetVector(weight_length, sizeof(data_type), this->dev_A, 1, this->weight_matrix[layer_index], 1); // W(t + 1) <- dev_A
			break;
		case Optimizer::Adam:
			// m(t + 1) = β1 * m(t) + (1 - β1) * ∇Loss(W(t))
			// v(t + 1) = β2 * v(t) + (1 - β2) * ∇Loss(W(t))²
			// m_hat = m(t + 1) / (1 - β1^t)
			// v_hat = v(t + 1) / (1 - β2^t)
			// W(t + 1) = W(t) - η * m_hat / (√(v_hat) + ε)
			scalar_alpha = 1.0 - this->beta1; // scalar_alpha = 1 - β1
			cublasSetVector(weight_length, sizeof(data_type), this->momentum_matrix[layer_index], 1, this->dev_D, 1); // dev_D <- m(t)
			cublasDscal(this->handle, weight_length, &this->beta1, this->dev_D, 1); // dev_D = β1 * m(t)
			cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_D, 1); // dev_D = (1 - β1) * ∇Loss(W(t)) + dev_D

			cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->result_C, 1); // result_C <- ∇Loss(W(t))
			for (size_t p = 0; p < weight_length; p++) this->result_C[p] = (1 - this->beta2) * this->result_C[p] * this->result_C[p]; // result_C = (1 - β2) * ∇Loss(W(t))²
			cublasSetVector(weight_length, sizeof(data_type), this->squared_gradient_matrix[layer_index], 1, this->dev_C, 1); // dev_C <- v(t)
			cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_B, 1); // dev_B <- result_C
			cublasDaxpy(this->handle, weight_length, &this->beta2, this->dev_C, 1, this->dev_B, 1); // dev_B = β2 * v(t) + dev_B

			cublasGetVector(weight_length, sizeof(data_type), this->dev_D, 1, this->momentum_matrix[layer_index], 1); // m(t + 1) <- dev_D
			cublasGetVector(weight_length, sizeof(data_type), this->dev_B, 1, this->squared_gradient_matrix[layer_index], 1); // v(t + 1) <- dev_B

			for (size_t p = 0; p < weight_length; p++) this->result_C[p] = 0; // result_C = {0,}

			scalar_alpha = 1.0 / (1.0 - pow(this->beta1, t)); // scalar_alpha = 1 / (1 - β1^t)
			cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_C, 1); // dev_C <- result_C
			cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_D, 1, this->dev_C, 1); // dev_C(m_hat) = 1 / (1 - β1^t) * dev_D + {0,}

			scalar_alpha = 1.0 / (1.0 - pow(this->beta2, t)); // scalar_alpha = 1 / (1 - β2^t)
			cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_D, 1); // dev_D <- result_C
			cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_B, 1, this->dev_D, 1); // dev_D(v_hat) = 1 / (1 - β2^t) * dev_B + {0,}

			cublasGetVector(weight_length, sizeof(data_type), this->dev_C, 1, this->result_C, 1); // result_C <- dev_C(m_hat)
			cublasGetVector(weight_length, sizeof(data_type), this->dev_D, 1, this->host_memory, 1); // host_memory <- dev_D(v_hat)
			for (size_t p = 0; p < weight_length; p++) this->result_C[p] = this->result_C[p] / (sqrt(this->host_memory[p]) + this->epsilon); // result_C = m_hat / (√(v_hat) + ε)

			cublasSetVector(weight_length, sizeof(data_type), this->weight_matrix[layer_index], 1, this->dev_D, 1); // dev_D <- W(t)
			cublasSetVector(weight_length, sizeof(data_type), this->result_C, 1, this->dev_C, 1); // dev_C <- result_C

			scalar_alpha = (-1) * this->learning_rate; // scalar_alpha = -1 * η
			cublasDaxpy(this->handle, weight_length, &scalar_alpha, this->dev_C, 1, this->dev_D, 1); // dev_D = (-1 * η) * dev_C + dev_D
			cublasGetVector(weight_length, sizeof(data_type), this->dev_D, 1, this->weight_matrix[layer_index], 1); // W(t + 1) <- dev_D
			break;
		default:
			printf("알 수 없는 최적화 기법. 프로그램 종료.\n");
			exit(1);
			break;
		}
		if (debug_mode) Functions::printArray("갱신 후 가중치[i]", this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1], this->weight_matrix[layer_index], this->number_of_nodes[layer_index - 1]);
	}
}

void bpl::Model::version()
{
	printf("%s\n", this->version_info.c_str());
}

void bpl::Model::clearModel()
{
	this->number_of_hidden_layer = 0;
	this->number_of_nodes = vector<size_t>();
	this->learning_rate = 0.001;
	this->optimizer = Optimizer::GD;
	this->beta1 = 0.9;
	this->beta2 = 0.999;
	this->loss_function = LossFunction::MSE;
	this->weight_init = WeightInitMethod::UniformDistribution;
	this->test_layer = vector<vector<data_type>>();
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->bias_matrix_no_batch[layer_index]);
	free(this->bias_matrix_no_batch);
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->weight_matrix[layer_index]);
	free(this->weight_matrix);
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->momentum_matrix[layer_index]);
	free(this->momentum_matrix);
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->squared_gradient_matrix[layer_index]);
	free(this->squared_gradient_matrix);
	this->active_method_matrix = vector<ActiveFunction>();
	this->is_prepared = false;
	this->is_loaded = false;
}

void bpl::Model::loadModel(string file_name)
{
	this->is_loaded = false;
	if (this->input_layer.size() == 0) errorHandling("loadModel", ErrorType::InputDataError);
	this->clearModel();

	FILE* pFile = nullptr;

	if (fopen_s(&pFile, file_name.c_str(), "r") != 0) { errorHandling("loadModel", ErrorType::FileOpenError, file_name); return; }
	fscanf_s(pFile, "%zu %zu", &this->number_of_input_node, &this->number_of_hidden_layer); // 모델 구조, 가중치 읽기

	this->number_of_nodes = vector<size_t>(this->number_of_hidden_layer);
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		fscanf_s(pFile, "%zu", &this->number_of_nodes[layer_index]);
		if (this->number_of_nodes[layer_index] > this->max_nodes_length) this->max_nodes_length = this->number_of_nodes[layer_index];
	}

	this->active_method_matrix = vector<ActiveFunction>(this->number_of_hidden_layer);
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		fscanf_s(pFile, "%d", &this->active_method_matrix[layer_index]);
	}

	fscanf_s(pFile, "%d %d %d %lf", &this->weight_init, &this->optimizer, &this->loss_function, &this->learning_rate);

	/* 메모리 할당 */
	// 가중치 행렬 메모리 할당
	this->weight_matrix = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->weight_matrix == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "weight_matrix"); return; }
	this->weight_matrix[0] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[0] * this->number_of_input_node); // input_layer -> hidden_layer[0]
	if (this->weight_matrix[0] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "weight_matrix"); return; }
	for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->weight_matrix[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1]);
		if (this->weight_matrix[layer_index] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "weight_matrix"); return; }
	}

	// no_batch 편향 행렬 메모리 할당
	this->bias_matrix_no_batch = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->bias_matrix_no_batch == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "bias_matrix_no_batch"); return; }
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->bias_matrix_no_batch[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index]);
		if (this->bias_matrix_no_batch[layer_index] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "bias_matrix_no_batch"); return; }
	}

	// velocity 행렬 메모리 할당
	this->momentum_matrix = (data_type**)calloc(this->number_of_hidden_layer, sizeof(data_type*));
	if (this->momentum_matrix == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "momentum_matrix"); return; }
	this->momentum_matrix[0] = (data_type*)calloc(this->number_of_nodes[0] * this->number_of_input_node, sizeof(data_type)); // input_layer -> hidden_layer[0]
	if (this->momentum_matrix[0] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "momentum_matrix"); return; }
	for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->momentum_matrix[layer_index] = (data_type*)calloc(this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1], sizeof(data_type));
		if (this->momentum_matrix[layer_index] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "momentum_matrix"); return; }
	}

	// squared_gradient 행렬 메모리 할당
	this->squared_gradient_matrix = (data_type**)calloc(this->number_of_hidden_layer, sizeof(data_type*));
	if (this->squared_gradient_matrix == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "squared_gradient_matrix"); return; }
	this->squared_gradient_matrix[0] = (data_type*)calloc(this->number_of_nodes[0] * this->number_of_input_node, sizeof(data_type)); // input_layer -> hidden_layer[0]
	if (this->squared_gradient_matrix[0] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "squared_gradient_matrix"); return; }
	for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->squared_gradient_matrix[layer_index] = (data_type*)calloc(this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1], sizeof(data_type));
		if (this->squared_gradient_matrix[layer_index] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "squared_gradient_matrix"); return; }
	}

	/* 가중치, 편향, 모멘텀, 기울기 행렬 불러오기 */
	for (size_t output_index = 0; output_index < this->number_of_nodes[0]; output_index++) // 현재 레이어 노드수 * 입력값 | + 1: bias | weight_matrix, bias_matrix_no_batch
	{
		for (size_t input_index = 0; input_index < this->number_of_input_node; input_index++) fscanf_s(pFile, "%lf", &this->weight_matrix[0][output_index * this->number_of_input_node + input_index]);
		fscanf_s(pFile, "%lf", &this->bias_matrix_no_batch[0][output_index]);
	}
	for (size_t p = 0; p < this->number_of_nodes[0] * this->number_of_input_node; p++) // 현재 레이어 노드수 * 입력값 | momentum_matrix
	{
		fscanf_s(pFile, "%lf", &this->momentum_matrix[0][p]);
	}
	for (size_t p = 0; p < this->number_of_nodes[0] * this->number_of_input_node; p++) // 현재 레이어 노드수 * 입력값 | squared_gradient_matrix
	{
		fscanf_s(pFile, "%lf", &this->squared_gradient_matrix[0][p]);
	}
	for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		for (size_t output_index = 0; output_index < this->number_of_nodes[layer_index]; output_index++) // 현재 레이어 노드수 * 이전 레이어 노드 수 | + 1: bias | weight_matrix, bias_matrix_no_batch
		{
			for (size_t input_index = 0; input_index < this->number_of_nodes[layer_index - 1]; input_index++) fscanf_s(pFile, "%lf", &this->weight_matrix[layer_index][output_index * this->number_of_nodes[layer_index - 1] + input_index]);
			fscanf_s(pFile, "%lf", &this->bias_matrix_no_batch[layer_index][output_index]);
		}
		for (size_t p = 0; p < this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1]; p++) // 현재 레이어 노드수 * 이전 레이어 노드 수 | momentum_matrix
		{
			fscanf_s(pFile, "%lf", &this->momentum_matrix[layer_index][p]);
		}
		for (size_t p = 0; p < this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1]; p++) // 현재 레이어 노드수 * 이전 레이어 노드 수 | squared_gradient_matrix
		{
			fscanf_s(pFile, "%lf", &this->squared_gradient_matrix[layer_index][p]);
		}
	}

	if (this->debug_mode)
	{
		printf("weight matrix\n");
		for (size_t i = 0; i < this->number_of_hidden_layer; i++)
		{
			if (i == 0)
			{
				for (size_t j = 0; j < this->number_of_nodes[0]; j++)
				{
					for (size_t k = 0; k < this->number_of_input_node; k++) printf("%lf ", this->weight_matrix[0][j * this->number_of_input_node + k]);
					printf("\n");
				}
			}
			else
			{
				for (size_t j = 0; j < this->number_of_nodes[i]; j++)
				{
					for (size_t k = 0; k < this->number_of_nodes[i - 1]; k++) printf("%lf ", this->weight_matrix[i][j * this->number_of_nodes[i - 1] + k]);
					printf("\n");
				}
			}
			printf("\n");
		}
		printf("bias matrix\n");
		for (size_t i = 0; i < this->number_of_hidden_layer; i++)
		{
			for (size_t j = 0; j < this->number_of_nodes[i]; j++) printf("%lf ", this->bias_matrix_no_batch[i][j]);
			printf("\n");
		}
		printf("\n");
		printf("momentum matrix\n");
		for (size_t i = 0; i < this->number_of_hidden_layer; i++)
		{
			if (i == 0)
			{
				for (size_t j = 0; j < this->number_of_nodes[0]; j++)
				{
					for (size_t k = 0; k < this->number_of_input_node; k++) printf("%lf ", this->momentum_matrix[0][j * this->number_of_input_node + k]);
					printf("\n");
				}
			}
			else
			{
				for (size_t j = 0; j < this->number_of_nodes[i]; j++)
				{
					for (size_t k = 0; k < this->number_of_nodes[i - 1]; k++) printf("%lf ", this->momentum_matrix[i][j * this->number_of_nodes[i - 1] + k]);
					printf("\n");
				}
			}
			printf("\n");
		}
		printf("squared gradient matrix\n");
		for (size_t i = 0; i < this->number_of_hidden_layer; i++)
		{
			if (i == 0)
			{
				for (size_t j = 0; j < this->number_of_nodes[0]; j++)
				{
					for (size_t k = 0; k < this->number_of_input_node; k++) printf("%lf ", this->squared_gradient_matrix[0][j * this->number_of_input_node + k]);
					printf("\n");
				}
			}
			else
			{
				for (size_t j = 0; j < this->number_of_nodes[i]; j++)
				{
					for (size_t k = 0; k < this->number_of_nodes[i - 1]; k++) printf("%lf ", this->squared_gradient_matrix[i][j * this->number_of_nodes[i - 1] + k]);
					printf("\n");
				}
			}
			printf("\n");
		}
	}

	fclose(pFile);
	this->is_loaded = true;
}

void bpl::Model::addDenseLayer(int node_count, ActiveFunction active_function)
{
	if (this->is_prepared || this->is_loaded) errorHandling("addLayer", ErrorType::ModelModifiedError);

	if (node_count > this->max_nodes_length) this->max_nodes_length = node_count;
	this->number_of_hidden_layer++;
	this->number_of_nodes.push_back(node_count);
	this->active_method_matrix.push_back(active_function);
}

void bpl::Model::addFlattenLayer()
{
}

void bpl::Model::addDropoutLayer(float dropout_rate)
{

}

void bpl::Model::readInputData(const char* file_name, bool print_input_data)
{
	this->dataset_size = 0;
	this->target = vector<vector<data_type>>();
	this->input_layer = vector<vector<data_type>>();

	FILE* pFile = nullptr;
	size_t input_length = 0; // 입력값 개수(노드 수)
	size_t target_length = 0; // 출력값(정답) 개수(노드 수)
	size_t sum_of_count = 0; // 전체 파일 입력 개수 | 파일 전체 값 % (입력값 + 출력값) == 0 ? pass : error

	data_type buffer; // 입력 버퍼

	if (fopen_s(&pFile, file_name, "r") != 0) { errorHandling("readInputData", ErrorType::FileOpenError, file_name); return; }
	fscanf_s(pFile, "%zu %zu", &input_length, &target_length); // 입력 데이터 구조 읽기

	vector<data_type> input_line(input_length); // 학습데이터 입력 레이어 1개
	vector<data_type> target_line(target_length);  // 학습데이터 실제값 레이어 1개
	size_t cols = 0; // 한 줄에 입력받는 값 인덱스
	size_t cols_length = input_length + target_length; // 한 줄에 최대 입력 수
	while (fscanf_s(pFile, "%lf", &buffer) != EOF)
	{
		sum_of_count++;
		if (cols < input_length) input_line[cols] = buffer;
		else target_line[cols - input_length] = buffer;

		if (cols == cols_length - 1)
		{
			this->input_layer.push_back(input_line);
			this->target.push_back(target_line);
			this->dataset_size++;
			cols = 0;
		}
		else cols++;
	}
	fclose(pFile);

	if (sum_of_count == 0) errorHandling("readInputData", ErrorType::NoCotentError, file_name);
	if (sum_of_count % cols_length != 0) errorHandling("readInputData", ErrorType::FileContentError, "전체 데이터가 (입력값 + 출력값)으로 나누어 떨어지지 않음.");
	this->number_of_input_node = input_length;

	if (print_input_data)
	{
		size_t dataset_size = this->target.size(); // = this->input_layer.size()

		for (size_t dataset_index = 0; dataset_index < dataset_size; dataset_index++)
		{
			printf("DataSet[%zu]: { Input: [", dataset_index);
			for (size_t i = 0; i < this->input_layer[dataset_index].size(); i++)
			{
				if (i < (this->input_layer[dataset_index].size() - 1)) printf("%.2lf, ", this->input_layer[dataset_index][i]);
				else printf("%.2lf]", this->input_layer[dataset_index][i]);
			}
			printf(", Target: [");
			for (size_t i = 0; i < this->target[dataset_index].size(); i++)
			{
				if (i < (this->target[dataset_index].size() - 1)) printf("%.2lf, ", this->target[dataset_index][i]);
				else printf("%.2lf]", this->target[dataset_index][i]);
			}
			printf(" }\n");
		}
	}
}

void bpl::Model::saveModel(const char* file_name)
{
	if (!this->is_prepared && !this->is_loaded) errorHandling("saveModel", ErrorType::LoadModelOrPrepareError);

	FILE* pFile = nullptr;
	if (fopen_s(&pFile, file_name, "w") != 0) { errorHandling("saveModel", ErrorType::FileOpenError, file_name); return; }
	fprintf(pFile, "%zu %zu\n", this->number_of_input_node, this->number_of_hidden_layer);
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		if (layer_index == (this->number_of_hidden_layer - 1)) fprintf(pFile, "%zu", this->number_of_nodes[layer_index]);
		else fprintf(pFile, "%zu ", this->number_of_nodes[layer_index]);
	}
	fprintf(pFile, "\n");
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		if (layer_index == (this->number_of_hidden_layer - 1)) fprintf(pFile, "%d", this->active_method_matrix[layer_index]);
		else fprintf(pFile, "%d ", this->active_method_matrix[layer_index]);
	}
	fprintf(pFile, "\n");
	fprintf(pFile, "%d %d %d %.4lf", this->weight_init, this->optimizer, this->loss_function, this->learning_rate);
	fprintf(pFile, "\n");
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		for (size_t matrix_m = 0; matrix_m < this->number_of_nodes[layer_index]; matrix_m++) // weight_matrix
		{
			if (layer_index == 0)
			{
				for (size_t matrix_n = 0; matrix_n < this->number_of_input_node; matrix_n++) fprintf(pFile, "%lf ", this->weight_matrix[0][matrix_m * this->number_of_input_node + matrix_n]);
				fprintf(pFile, "%lf", this->bias_matrix_no_batch[0][matrix_m]); // bias_matrix
			}
			else
			{
				for (size_t matrix_n = 0; matrix_n < this->number_of_nodes[layer_index - 1]; matrix_n++) fprintf(pFile, "%lf ", this->weight_matrix[layer_index][matrix_m * this->number_of_nodes[layer_index - 1] + matrix_n]);
				fprintf(pFile, "%lf", this->bias_matrix_no_batch[layer_index][matrix_m]); // bias_matrix
			}
			fprintf(pFile, "\n");
		}
		fprintf(pFile, "\n");
		for (size_t matrix_m = 0; matrix_m < this->number_of_nodes[layer_index]; matrix_m++) // momentum_matrix
		{
			if (layer_index == 0)
			{
				for (size_t matrix_n = 0; matrix_n < this->number_of_input_node - 1; matrix_n++) fprintf(pFile, "%lf ", this->momentum_matrix[0][matrix_m * this->number_of_input_node + matrix_n]);
				fprintf(pFile, "%lf", this->momentum_matrix[0][matrix_m * this->number_of_input_node + this->number_of_input_node - 1]);
			}
			else
			{
				for (size_t matrix_n = 0; matrix_n < this->number_of_nodes[layer_index - 1] - 1; matrix_n++) fprintf(pFile, "%lf ", this->momentum_matrix[layer_index][matrix_m * this->number_of_nodes[layer_index - 1] + matrix_n]);
				fprintf(pFile, "%lf", this->momentum_matrix[layer_index][matrix_m * this->number_of_nodes[layer_index - 1] + this->number_of_nodes[layer_index - 1] - 1]);
			}
			fprintf(pFile, "\n");
		}
		fprintf(pFile, "\n");
		for (size_t matrix_m = 0; matrix_m < this->number_of_nodes[layer_index]; matrix_m++) // sqaured_gradient_matrix
		{
			if (layer_index == 0)
			{
				for (size_t matrix_n = 0; matrix_n < this->number_of_input_node - 1; matrix_n++) fprintf(pFile, "%lf ", this->squared_gradient_matrix[0][matrix_m * this->number_of_input_node + matrix_n]);
				fprintf(pFile, "%lf", this->squared_gradient_matrix[0][matrix_m * this->number_of_input_node + this->number_of_input_node - 1]);
			}
			else
			{
				for (size_t matrix_n = 0; matrix_n < this->number_of_nodes[layer_index - 1] - 1; matrix_n++) fprintf(pFile, "%lf ", this->squared_gradient_matrix[layer_index][matrix_m * this->number_of_nodes[layer_index - 1] + matrix_n]);
				fprintf(pFile, "%lf", this->squared_gradient_matrix[layer_index][matrix_m * this->number_of_nodes[layer_index - 1] + this->number_of_nodes[layer_index - 1] - 1]);
			}
			fprintf(pFile, "\n");
		}
		fprintf(pFile, "\n");
	}
	fclose(pFile);
}

void bpl::Model::prepare(hyper_parameters params)
{
	this->is_prepared = false;
	if (this->input_layer.size() == 0) errorHandling("prepare", ErrorType::InputDataError);
	if (this->number_of_hidden_layer == 0) errorHandling("prepare", ErrorType::NoLayerSetError);
	if (this->number_of_nodes.back() != this->target[0].size()) errorHandling("prepare", ErrorType::OuputLengthMatchError);

	/* 메모리 할당 */
	// 가중치 행렬 메모리 할당
	this->weight_matrix = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->weight_matrix == NULL) { errorHandling("prepare", ErrorType::InSufficientMemory, "weight_matrix"); return; }
	this->weight_matrix[0] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[0] * this->number_of_input_node); // input_layer -> hidden_layer[0]
	if (this->weight_matrix[0] == NULL) { errorHandling("prepare", ErrorType::InSufficientMemory, "weight_matrix"); return; }
	for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->weight_matrix[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1]);
		if (this->weight_matrix[layer_index] == NULL) { errorHandling("prepare", ErrorType::InSufficientMemory, "weight_matrix"); return; }
	}

	// no_batch 편향 행렬 메모리 할당
	this->bias_matrix_no_batch = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->bias_matrix_no_batch == NULL) { errorHandling("prepare", ErrorType::InSufficientMemory, "bias_matrix_no_batch"); return; }
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->bias_matrix_no_batch[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index]);
		if (this->bias_matrix_no_batch[layer_index] == NULL) { errorHandling("prepare", ErrorType::InSufficientMemory, "bias_matrix_no_batch"); return; }
	}

	// velocity 행렬 메모리 할당
	this->momentum_matrix = (data_type**)calloc(this->number_of_hidden_layer, sizeof(data_type*));
	if (this->momentum_matrix == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "momentum_matrix"); return; }
	this->momentum_matrix[0] = (data_type*)calloc(this->number_of_nodes[0] * this->number_of_input_node, sizeof(data_type)); // input_layer -> hidden_layer[0]
	if (this->momentum_matrix[0] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "momentum_matrix"); return; }
	for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->momentum_matrix[layer_index] = (data_type*)calloc(this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1], sizeof(data_type));
		if (this->momentum_matrix[layer_index] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "momentum_matrix"); return; }
	}

	// squared_gradient 행렬 메모리 할당
	this->squared_gradient_matrix = (data_type**)calloc(this->number_of_hidden_layer, sizeof(data_type*));
	if (this->squared_gradient_matrix == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "squared_gradient_matrix"); return; }
	this->squared_gradient_matrix[0] = (data_type*)calloc(this->number_of_nodes[0] * this->number_of_input_node, sizeof(data_type)); // input_layer -> hidden_layer[0]
	if (this->squared_gradient_matrix[0] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "squared_gradient_matrix"); return; }
	for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->squared_gradient_matrix[layer_index] = (data_type*)calloc(this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1], sizeof(data_type));
		if (this->squared_gradient_matrix[layer_index] == NULL) { errorHandling("loadModel", ErrorType::InSufficientMemory, "squared_gradient_matrix"); return; }
	}

	/* 가중치 초기화 */
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<data_type> distu(0.0, 1.0); // 0.0 ~ 1.0
	normal_distribution<data_type> distn(0.5, 0.5); // 0.0 ~ 1.0
	switch (params.weight_init)
	{
	case WeightInitMethod::ZeroInitialize:
	{
		for (size_t offset = 0; offset < this->number_of_nodes[0] * this->number_of_input_node; offset++)
		{
			this->weight_matrix[0][offset] = 0.0;
		}
		for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
		{
			for (size_t offset = 0; offset < this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1]; offset++)
			{
				this->weight_matrix[layer_index][offset] = 0.0;
			}
		}
		break;
	}
	case WeightInitMethod::NormalDistribution:
	{
		for (size_t offset = 0; offset < this->number_of_nodes[0] * this->number_of_input_node; offset++)
		{
			this->weight_matrix[0][offset] = distn(gen);
		}
		for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
		{
			for (size_t offset = 0; offset < this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1]; offset++)
			{
				this->weight_matrix[layer_index][offset] = distn(gen);
			}
		}
		break;
	}
	case WeightInitMethod::UniformDistribution:
	{
		for (size_t offset = 0; offset < this->number_of_nodes[0] * this->number_of_input_node; offset++)
		{
			this->weight_matrix[0][offset] = distu(gen);
		}
		for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
		{
			for (size_t offset = 0; offset < this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1]; offset++)
			{
				this->weight_matrix[layer_index][offset] = distu(gen);
			}
		}
		break;
	}
	case WeightInitMethod::XavierNormalDistribution:
	{
		normal_distribution<data_type> xdistn0(0, sqrt(2.0 / (this->number_of_input_node + this->number_of_nodes[0])));
		for (size_t offset = 0; offset < this->number_of_nodes[0] * this->number_of_input_node; offset++)
		{
			this->weight_matrix[0][offset] = xdistn0(gen);
		}
		for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
		{
			normal_distribution<data_type> xdistn(0, sqrt(2.0 / (this->number_of_nodes[layer_index - 1] + this->number_of_nodes[layer_index])));
			for (size_t offset = 0; offset < this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1]; offset++)
			{
				this->weight_matrix[layer_index][offset] = xdistn(gen);
			}
		}
		break;
	}
	case WeightInitMethod::XavierUniformDistribution:
	{
		uniform_real_distribution<data_type> xdistu0(-sqrt(6.0 / (this->number_of_input_node + this->number_of_nodes[0])), sqrt(6.0 / (this->number_of_input_node + this->number_of_nodes[0])));
		for (size_t offset = 0; offset < this->number_of_nodes[0] * this->number_of_input_node; offset++)
		{
			this->weight_matrix[0][offset] = xdistu0(gen);
		}
		for (size_t layer_index = 1; layer_index < this->number_of_hidden_layer; layer_index++)
		{
			uniform_real_distribution<data_type> xdistu(-sqrt(6.0 / (this->number_of_nodes[layer_index - 1] + this->number_of_nodes[layer_index])), sqrt(6.0 / (this->number_of_nodes[layer_index - 1] + this->number_of_nodes[layer_index])));
			for (size_t offset = 0; offset < this->number_of_nodes[layer_index] * this->number_of_nodes[layer_index - 1]; offset++)
			{
				this->weight_matrix[layer_index][offset] = xdistu(gen);
			}
		}
		break;
	}
	case WeightInitMethod::He:
	{
		break;
	}
	default:
	{
		break;
	}
	}

	/* no_batch 편향 초기화 */
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		for (size_t p = 0; p < this->number_of_nodes[layer_index]; p++)
		{
			this->bias_matrix_no_batch[layer_index][p] = 0; // 0으로 초기화
		}
	}

	// debug_mode 가중치 행렬, 편향 행렬 출력
	if (this->debug_mode)
	{
		printf("weight matrix initialize\n");

		printf("bias matrix initialize\n");
	}


	/* 하이퍼 파라미터 입력 */
	this->weight_init = params.weight_init;
	this->learning_rate = params.learning_rate;
	this->loss_function = params.loss_function;
	this->optimizer = params.optimizer;
	this->beta1 = params.beta1;
	this->beta2 = params.beta2;

	this->is_prepared = true;
}

void bpl::Model::printModel()
{
	if (!this->is_prepared && !this->is_loaded) errorHandling("printModel", ErrorType::LoadModelOrPrepareError);
	size_t total_params = 0;

	printf("=====================================================================\n");
	printf("Layer                Input Shape       Output Shape      Param\n");
	printf("---------------------------------------------------------------------\n");
	printf("Input                ");
	printf("%-18zu", this->number_of_input_node);
	printf("%-18zu", this->number_of_input_node);
	printf("None");
	printf("\n---------------------------------------------------------------------\n");
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		if (layer_index == 0)
		{
			total_params += (this->number_of_input_node + 1) * this->number_of_nodes[layer_index];
			printf("Hidden %-14zu", layer_index);
			printf("%-18zu", this->number_of_input_node);
			printf("%-18zu", this->number_of_nodes[layer_index]);
			printf("%zu", (this->number_of_input_node + 1) * this->number_of_nodes[layer_index]);
		}
		else if (layer_index == this->number_of_hidden_layer - 1)
		{
			total_params += (this->number_of_nodes[layer_index - 1] + 1) * this->number_of_nodes[layer_index];
			printf("(Output)Hidden %-6zu", layer_index);
			printf("%-18zu", this->number_of_nodes[layer_index - 1]);
			printf("%-18zu", this->number_of_nodes[layer_index]);
			printf("%zu", (this->number_of_nodes[layer_index - 1] + 1) * this->number_of_nodes[layer_index]);
		}
		else
		{
			total_params += (this->number_of_nodes[layer_index - 1] + 1) * this->number_of_nodes[layer_index];
			printf("Hidden %-14zu", layer_index);
			printf("%-18zu", this->number_of_nodes[layer_index - 1]);
			printf("%-18zu", this->number_of_nodes[layer_index]);
			printf("%zu", (this->number_of_nodes[layer_index - 1] + 1) * this->number_of_nodes[layer_index]);
		}
		printf("\n---------------------------------------------------------------------\n");
	}
	printf("Total Params: %zu\n", total_params);
	printf("DataSet: %zu\n", this->dataset_size);
	printf("Loss Function: ");
	switch (this->loss_function)
	{
	case LossFunction::MSE: printf("MSE"); break;
	case LossFunction::BinaryCrossEntropyLoss: printf("Binary Cross Entropy Loss"); break;
	case LossFunction::CategoricalCrossEntropyLoss: printf("Categorical Cross Entropy Loss"); break;
	case LossFunction::SparseCrossEntropyLoss: printf("Sparse Cross Entropy Loss"); break;
	case LossFunction::HingeLoss: printf("Hinge Loss"); break;
	default: break;
	}
	printf("\nOptimizer: ");
	switch (this->optimizer)
	{
	case Optimizer::GD: printf("GD"); break;
	case Optimizer::SGD: printf("SGD"); break;
	case Optimizer::Momentum: printf("Momentum"); break;
	case Optimizer::AdaGrad: printf("AdaGrad"); break;
	case Optimizer::RMSProp: printf("RMSProp"); break;
	case Optimizer::Adam: printf("Adam"); break;
	default: break;
	}
	printf("\n=====================================================================\n");
}

void bpl::Model::learning(size_t epoch, size_t batch_size, early_stopping_parameters early_stopping, learning_verbose_parameters verbose_option)
{
	if (!this->is_prepared && !this->is_loaded) errorHandling("learning", ErrorType::LoadModelOrPrepareError);

	// 예측값 행렬 메모리 할당
	this->out_hidden_layer = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->out_hidden_layer == NULL) { errorHandling("learning", ErrorType::InSufficientMemory, "out_hidden_layer"); return; }
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->out_hidden_layer[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index] * batch_size);
		if (this->out_hidden_layer[layer_index] == NULL) { errorHandling("learning", ErrorType::InSufficientMemory, "out_hidden_layer"); return; }
	}

	// 델타값 행렬 메모리 할당
	this->delta_hidden_layer = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->delta_hidden_layer == NULL) { errorHandling("learning", ErrorType::InSufficientMemory, "delta_hidden_layer"); return; }
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->delta_hidden_layer[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index] * batch_size);
		if (this->delta_hidden_layer[layer_index] == NULL) { errorHandling("learning", ErrorType::InSufficientMemory, "delta_hidden_layer"); return; }
	}

	// 편향 행렬 메모리 할당
	this->bias_matrix = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->bias_matrix == NULL) { errorHandling("learning", ErrorType::InSufficientMemory, "bias_matrix"); return; }
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->bias_matrix[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index] * batch_size);
		if (this->bias_matrix[layer_index] == NULL) { errorHandling("learning", ErrorType::InSufficientMemory, "bias_matrix"); return; }
	}
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		for (size_t batch_index = 0; batch_index < batch_size; batch_index++) // batch_size만큼 반복
		{
			for (size_t p = 0; p < this->number_of_nodes[layer_index]; p++)
			{
				this->bias_matrix[layer_index][batch_index * this->number_of_nodes[layer_index] + p] = this->bias_matrix_no_batch[layer_index][p];
			}
		}
	}

	// CUBLAS 반환값 행렬 메모리 할당
	this->result_C = (data_type*)malloc(sizeof(data_type) * this->max_nodes_length * (this->max_nodes_length + batch_size));
	if (this->result_C == NULL) { errorHandling("learning", ErrorType::InSufficientMemory, "result_C"); return; }

	// host_memory 메모리 할당
	this->host_memory = (data_type*)malloc(sizeof(data_type) * this->max_nodes_length * this->max_nodes_length);
	if (this->host_memory == NULL) { errorHandling("learning", ErrorType::InSufficientMemory, "result_C"); return; }

	size_t dataset_length = this->input_layer.size();
	data_type loss = 0.0; // 매 학습 오차값
	data_type* loss_ptr = nullptr;
	data_type sum_of_loss = 0.0; // 전체 데이터 오차값
	data_type current_loss = 0.0; // 현재 오차값
	data_type prev_loss = 0.0; // 이전 오차값
	data_type best_loss = 0.0; // 최선의 오차값
	data_type accuracy = 0.0; // 매 학습 정확도
	data_type* accuracy_ptr = nullptr;
	vector<vector<data_type>> best_weight; // 최선의 가중치
	unsigned int patience_stack = 0;
	unsigned int progress = 0; // 학습 진행도: 1 ~20 | 100 / 20 = 5
	data_type percentage = 0.0; // 학습 진행률
	unsigned int percentage_i = 0; // 학습 진행률 / progress
	size_t mini_batch_length = static_cast<size_t>(ceil((data_type)dataset_length / (data_type)batch_size)); // mini batch 개수
	chrono::system_clock::time_point start;
	chrono::duration<data_type> end;
	data_type duration = 0.0; // 1 step 학습 시간(초)
	unsigned int duration_milli; // milliseconds
	unsigned int duration_micro; // microseconds

	if (verbose_option.error_verbose) loss_ptr = &loss;

	FILE* pFile = nullptr;
	if (verbose_option.write_file) if (fopen_s(&pFile, verbose_option.write_file_name, "w") != 0) { printf("Fail to learning file open.\n"); return; }

	checkCUBLAS(cublasCreate(&this->handle));
	checkCUDA(cudaMalloc((void**)&this->dev_A, this->max_nodes_length * this->max_nodes_length * sizeof(data_type)));
	checkCUDA(cudaMalloc((void**)&this->dev_B, this->max_nodes_length * (this->max_nodes_length + batch_size) * sizeof(data_type)));
	checkCUDA(cudaMalloc((void**)&this->dev_C, this->max_nodes_length * (this->max_nodes_length + batch_size) * sizeof(data_type)));
	checkCUDA(cudaMalloc((void**)&this->dev_D, this->max_nodes_length * this->max_nodes_length * sizeof(data_type)));

	for (int iteration = 1; iteration < epoch + 1; iteration++)
	{
		if (verbose_option.error_verbose)
		{
			printf("Epoch %d/%zu\n", iteration, epoch);
			sum_of_loss = 0.0;
			progress = 0;
			percentage = 0.0;
		}

		start = chrono::system_clock::now();
		for (size_t dataset_index = 0; dataset_index < dataset_length; dataset_index += batch_size)
		{
			chrono::system_clock::time_point start_prop;
			if (dataset_index + batch_size > dataset_length) // 데이터셋이 배치 사이즈로 나누어 떨어지지 않은 때 마지막 배치
			{
				start_prop = chrono::system_clock::now();
				this->forwardPropagation(dataset_index, dataset_length - dataset_index, loss_ptr, accuracy_ptr);
				end = chrono::system_clock::now() - start_prop;
				if (this->verbose_time) printf("\n순전파 시간: %lf\n", end.count());
				start_prop = chrono::system_clock::now();
				this->backPropagation(dataset_index, dataset_length - dataset_index, iteration);
				end = chrono::system_clock::now() - start_prop;
				if (this->verbose_time) printf("\n역전파 시간: %lf\n", end.count());
			}
			else
			{
				start_prop = chrono::system_clock::now();
				this->forwardPropagation(dataset_index, batch_size, loss_ptr, accuracy_ptr);
				end = chrono::system_clock::now() - start_prop;
				if (this->verbose_time) printf("\n순전파 시간: %lf\n", end.count());
				start_prop = chrono::system_clock::now();
				this->backPropagation(dataset_index, batch_size, iteration);
				end = chrono::system_clock::now() - start_prop;
				if (this->verbose_time) printf("\n역전파 시간: %lf\n", end.count());
			}

			if (verbose_option.error_verbose)
			{
				sum_of_loss += loss;
				percentage += 100.0 / mini_batch_length;
				percentage_i = percentage / 5.0;
				printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
				printf("%zu/%zu [", dataset_index / batch_size + 1, mini_batch_length);
				for (unsigned int i = 0; i < percentage_i; i++)
				{
					if (i == (percentage_i - 1)) printf(">");
					else printf("=");
				}
				progress = percentage_i;
			}
		}
		if (verbose_option.error_verbose) printf("\b]");
		end = chrono::system_clock::now() - start;

		if ((early_stopping.patience > 0 && iteration > early_stopping.start_from_epoch) || verbose_option.error_verbose || verbose_option.write_file)
		{
			prev_loss = current_loss;
			current_loss = sum_of_loss / mini_batch_length;
		}

		if (verbose_option.write_file) fprintf(pFile, "%lf\n", current_loss);

		if (verbose_option.error_verbose)
		{
			duration = end.count();
			duration_milli = (unsigned int)(duration * 1000) % 1000;
			duration_micro = (unsigned int)(duration * 1000000) % 1000;
			if ((unsigned int)duration > 0) printf(" %ds %dms/step", (unsigned int)duration, duration_milli);
			else printf(" %dms %dμs/step", duration_milli, duration_micro);
			if (verbose_option.error_verbose) printf(" - loss: %lf", current_loss);
			printf("\n");
		}

		if (early_stopping.patience > 0 && iteration > early_stopping.start_from_epoch)
		{
			if ((prev_loss - current_loss) > early_stopping.min_loss) patience_stack = 0;
			else patience_stack++;

			//if(current_loss < 0) { printf("Learning Stopped.\n"); break; } // 학습 종료
			if (patience_stack >= early_stopping.patience) { printf("Learning Stopped.\n"); break; } // 학습 종료
		}
	}
	if (verbose_option.write_file) fclose(pFile);

	checkCUDA(cudaFree(this->dev_A));
	checkCUDA(cudaFree(this->dev_B));
	checkCUDA(cudaFree(this->dev_C));
	checkCUDA(cudaFree(this->dev_D));
	checkCUBLAS(cublasDestroy(this->handle));

	/* 메모리 해제 */
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->out_hidden_layer[layer_index]);
	free(this->out_hidden_layer);
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->delta_hidden_layer[layer_index]);
	free(this->delta_hidden_layer);
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->bias_matrix[layer_index]);
	free(this->bias_matrix);
	free(this->result_C);
	free(this->host_memory);
}

vector<data_type> bpl::Model::predict(vector<data_type> testdata)
{
	if (!this->is_prepared && !this->is_loaded) errorHandling("predict", ErrorType::LoadModelOrPrepareError);
	if (this->number_of_input_node != testdata.size()) errorHandling("predict", ErrorType::TestLengthMatchError);

	// no_batch 예측값 행렬 메모리 할당
	this->out_hidden_layer_no_batch = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->out_hidden_layer_no_batch == NULL) { errorHandling("predict", ErrorType::InSufficientMemory, "out_hidden_layer_no_batch"); return vector<data_type>(); }
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->out_hidden_layer_no_batch[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index]);
		if (this->out_hidden_layer_no_batch[layer_index] == NULL) { errorHandling("predict", ErrorType::InSufficientMemory, "out_hidden_layer_no_batch"); return vector<data_type>(); }
	}

	checkCUBLAS(cublasCreate(&this->handle));
	this->forwardPropagationFromVector(testdata);
	checkCUBLAS(cublasDestroy(this->handle));

	vector<data_type> result(this->number_of_nodes.back());
	for (size_t p = 0; p < this->number_of_nodes.back(); p++) result[p] = this->out_hidden_layer_no_batch[this->number_of_hidden_layer - 1][p];

	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->out_hidden_layer_no_batch[layer_index]);
	free(this->out_hidden_layer_no_batch);

	return result;
}

vector<vector<data_type>> bpl::Model::predictFromFile(const char* test_file)
{
	if (!this->is_prepared && !this->is_loaded) errorHandling("predictFromFile", ErrorType::LoadModelOrPrepareError);
	this->readTestData(test_file);

	// no_batch 예측값 행렬 메모리 할당
	this->out_hidden_layer_no_batch = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->out_hidden_layer_no_batch == NULL) { errorHandling("predictFromFile", ErrorType::InSufficientMemory, "out_hidden_layer_no_batch"); return vector<vector<data_type>>(); }
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->out_hidden_layer_no_batch[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index]);
		if (this->out_hidden_layer_no_batch[layer_index] == NULL) { errorHandling("predictFromFile", ErrorType::InSufficientMemory, "out_hidden_layer_no_batch"); return vector<vector<data_type>>(); }
	}

	vector<vector<data_type>> result(this->test_layer.size(), vector<double>(this->number_of_nodes.back()));
	checkCUBLAS(cublasCreate(&this->handle));
	for (size_t dataset_index = 0; dataset_index < this->test_layer.size(); dataset_index++)
	{
		this->forwardPropagationFromVector(this->test_layer[dataset_index]);
		for (size_t p = 0; p < this->number_of_nodes.back(); p++) result[dataset_index][p] = this->out_hidden_layer_no_batch[this->number_of_hidden_layer - 1][p];
	}

	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->out_hidden_layer_no_batch[layer_index]);
	free(this->out_hidden_layer_no_batch);
	checkCUBLAS(cublasDestroy(this->handle));

	return result;
}

void bpl::Model::predictToFile(vector<data_type> testdata, const char* delimiter, const char* output_file, bool write_with_testdata)
{
	if (!this->is_prepared && !this->is_loaded) errorHandling("predictToFile", ErrorType::LoadModelOrPrepareError);
	if (this->number_of_input_node != testdata.size()) errorHandling("predictToFile", ErrorType::TestLengthMatchError);

	// no_batch 예측값 행렬 메모리 할당
	this->out_hidden_layer_no_batch = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->out_hidden_layer_no_batch == NULL) { errorHandling("predictToFile", ErrorType::InSufficientMemory, "out_hidden_layer_no_batch"); return; }
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->out_hidden_layer_no_batch[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index]);
		if (this->out_hidden_layer_no_batch[layer_index] == NULL) { errorHandling("predictToFile", ErrorType::InSufficientMemory, "out_hidden_layer_no_batch"); return; }
	}

	FILE* pFile = nullptr;
	if (fopen_s(&pFile, output_file, "w") != 0) { errorHandling("predictToFile", ErrorType::FileOpenError, output_file); return; }

	checkCUBLAS(cublasCreate(&this->handle));
	this->forwardPropagationFromVector(testdata);
	checkCUBLAS(cublasDestroy(this->handle));

	if (write_with_testdata) fprintf(pFile, "test%s", delimiter);
	fprintf(pFile, "output\n");
	if (write_with_testdata) for (size_t p = 0; p < testdata.size(); p++) fprintf(pFile, "%lf%s", testdata[p], delimiter);
	for (size_t p = 0; p < this->number_of_nodes.back(); p++)
	{
		if (p == this->number_of_nodes.back() - 1) fprintf(pFile, "%lf", this->out_hidden_layer_no_batch[this->number_of_hidden_layer - 1][p]);
		else fprintf(pFile, "%lf%s", this->out_hidden_layer_no_batch[this->number_of_hidden_layer - 1][p], delimiter);
	}
	fclose(pFile);

	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->out_hidden_layer_no_batch[layer_index]);
	free(this->out_hidden_layer_no_batch);
}

void bpl::Model::predictToFileFromFile(const char* test_file, const char* delimiter, const char* output_file, bool write_with_testdata)
{
	if (!this->is_prepared && !this->is_loaded) errorHandling("predictToFileFromFile", ErrorType::LoadModelOrPrepareError);
	this->readTestData(test_file);

	// no_batch 예측값 행렬 메모리 할당
	this->out_hidden_layer_no_batch = (data_type**)malloc(sizeof(data_type*) * this->number_of_hidden_layer);
	if (this->out_hidden_layer_no_batch == NULL) { errorHandling("predictToFileFromFile", ErrorType::InSufficientMemory, "out_hidden_layer_no_batch"); return; }
	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++)
	{
		this->out_hidden_layer_no_batch[layer_index] = (data_type*)malloc(sizeof(data_type) * this->number_of_nodes[layer_index]);
		if (this->out_hidden_layer_no_batch[layer_index] == NULL) { errorHandling("predictToFileFromFile", ErrorType::InSufficientMemory, "out_hidden_layer_no_batch"); return; }
	}

	FILE* pFile = nullptr;
	if (fopen_s(&pFile, output_file, "w") != 0) { errorHandling("predictToFileFromFile", ErrorType::FileOpenError, output_file); return; }

	if (write_with_testdata) fprintf(pFile, "test%s", delimiter);
	fprintf(pFile, "output\n");

	checkCUBLAS(cublasCreate(&this->handle));
	for (size_t dataset_index = 0; dataset_index < this->test_layer.size(); dataset_index++)
	{
		this->forwardPropagationFromVector(this->test_layer[dataset_index]);

		if (write_with_testdata) for (size_t p = 0; p < this->test_layer[dataset_index].size(); p++) fprintf(pFile, "%lf%s", this->test_layer[dataset_index][p], delimiter);
		for (size_t p = 0; p < this->number_of_nodes.back(); p++)
		{
			if (p == this->number_of_nodes.back() - 1) fprintf(pFile, "%lf", this->out_hidden_layer_no_batch[this->number_of_hidden_layer - 1][p]);
			else fprintf(pFile, "%lf%s", this->out_hidden_layer_no_batch[this->number_of_hidden_layer - 1][p], delimiter);
		}
		fprintf(pFile, "\n");
	}
	fclose(pFile);

	for (size_t layer_index = 0; layer_index < this->number_of_hidden_layer; layer_index++) free(this->out_hidden_layer_no_batch[layer_index]);
	free(this->out_hidden_layer_no_batch);
	checkCUBLAS(cublasDestroy(this->handle));
}
