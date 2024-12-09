#include "Functions.h"

void bpl::Functions::printArray(const char* arr_name, int length, double* arr, int ld_arr)
{
	if (ld_arr < 1) { cout << " ** ld_arr�� 1���� ���� �� ����. **" << '\n'; return; }
	cout << arr_name << '\n';
	for (int i = 0; i < length; i++)
	{
		cout << arr[i] << ' ';
		if ((i + 1) % ld_arr == 0) cout << '\n';
	}
	cout << '\n';
}

vector<vector<double>> bpl::Functions::matrixMultiplication(vector<vector<double>> matrix1, vector<vector<double>> matrix2)
{
	size_t matrix1_m_length = matrix1.size(); // ����� 2����: ��
	size_t matrix1_n_length = matrix1[0].size(); // ����� 1����: ��
	size_t matrix2_n_length = matrix2.size(); // ����� 2����: ��
	size_t matrix2_r_length = matrix2[0].size(); // ����� 1����: ��
	vector<vector<double>> result(matrix1_m_length, vector<double>(matrix2_r_length)); // m * r
	double sum = 0;

	if (matrix1_m_length < 1 || matrix2_n_length < 1) { printf("����� ũ��� �ּ� 1 * 1 �̻��̿��� ��.\n"); return vector<vector<double>>(); }
	if (matrix1_n_length != matrix2_n_length) { printf("�� ����� ���� ���� �� ����� ���� ���� ��ġ���� ����.\n");  return vector<vector<double>>(); }

	for (size_t m = 0; m < matrix1_m_length; m++)
	{
		for (size_t r = 0; r < matrix2_r_length; r++) // = (k < matrix2_n_length) ������� ���� ������� ���� ����
		{
			for (size_t k = 0; k < matrix1_n_length; k++) sum += matrix1[m][k] * matrix2[k][r];
			result[m][r] = sum;
			sum = 0;
		}
	}
	return result;
}

double* bpl::Functions::matrixMultiplication(int m, int n, int k, double* A, double* B)
{
	double* result = (double*)malloc(sizeof(double) * m * n);
	if (result)
	{
		for (int idx_m = 0; idx_m < m; idx_m++)
		{
			for (int idx_n = 0; idx_n < n; idx_n++)
			{
				double sum = 0;
				for (int idx_k = 0; idx_k < k; idx_k++) sum += A[idx_m * k + idx_k] * B[idx_k * n + idx_n];
				result[idx_m * n + idx_n] = sum;
			}
		}
	}
	return result;
}

vector<int> bpl::Functions::oneHotEncoding(vector<double> data)
{
	double max = 0.0;
	int max_index = 0;
	vector<int> result = vector<int>(0);
	for (size_t i = 0; i < data.size(); i++)
	{
		if (data[i] > max)
		{
			max = data[i];
			max_index = i;
		}
	}
	result[max_index] = 1;
	return result;
}

void bpl::Functions::oneHotEncoding(double* data, int num_classes)
{
	double max = 0.0;
	int max_index = 0;
	for (int i = 0; i < num_classes; i++)
	{
		if (data[i] > max)
		{
			max = data[i];
			max_index = i;
		}
		data[i] = 0.0;
	}
	data[max_index] = 1.0;
}

double bpl::Functions::Sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double bpl::Functions::Sigmoid(double a, double b, double x)
{
	return 1.0 / (1.0 + exp(-a * x + b));
}

double bpl::Functions::tanh(double x)
{
	return 2 * Sigmoid(2 * x) - 1;
}

vector<double> bpl::Functions::Softmax(vector<double> x)
{
	double max = 0;
	double sum = 0;
	vector<double> result(x.size());
	for (size_t i = 0; i < x.size(); i++) if (max < x[i]) max = x[i]; // max
	for (size_t i = 0; i < x.size(); i++)
	{
		result[i] = exp(x[i] - max); // overflow ����
		sum += result[i]; // sum of exp(x[i] - max)
	}
	for (size_t i = 0; i < x.size(); i++)  result[i] /= sum; // exp(x[i] - max) / sum(x)

	return result;
}

void bpl::Functions::Softmax(double* x, int x_length)
{
	double max = 0;
	double sum = 0;
	double* result = (double*)malloc(sizeof(double) * x_length);
	if (!result) return;
	for (size_t i = 0; i < x_length; i++) if (max < x[i]) max = x[i]; // max
	for (size_t i = 0; i < x_length; i++)
	{
		result[i] = exp(x[i] - max); // overflow ����
		sum += result[i]; // sum of exp(x[i] - max)
	}
	for (size_t i = 0; i < x_length; i++)  x[i] = result[i] / sum; // exp(x[i] - max) / sum(x)
	free(result);
}

double bpl::Functions::ReLU(double x)
{
	return x > 0 ? x : 0.0;
}

double bpl::Functions::leakyReLU(double x)
{
	return x > 0 ? x : 0.01 * x;
}

double bpl::Functions::ELU(double x)
{
	return x > 0 ? x : exp(x) - 1;
}

double bpl::Functions::Swish(double x, double beta)
{
	return x * Sigmoid(beta * x);
}

double bpl::Functions::MSE(vector<double> target, vector<double> output)
{
	if (target.size() != output.size()) { printf("MSE Error: �������� �������� �迭 ũ�Ⱑ ��ġ���� ����.\n"); return 0.0; }
	size_t error_length = target.size(); // = output.size()
	double error = 0;
	for (size_t i = 0; i < error_length; i++) error += pow(target[i] - output[i], 2); // ��(target - output)��

	return error / error_length;
}

double bpl::Functions::MSE(double* target, double* output, int n)
{
	double error = 0;
	for (size_t i = 0; i < n; i++) error += pow(target[i] - output[i], 2); // ��(target - output)��

	return error / n;
}

double bpl::Functions::BinaryCrossEntropy(vector<double> target, vector<double> output)
{
	if (target.size() != output.size()) { printf("Binary CrossEntropyLoss Error: �������� �������� �迭 ũ�Ⱑ ��ġ���� ����.\n"); return 0.0; }
	size_t error_length = target.size(); // = output.size()
	return 0.0;
}

double bpl::Functions::CategoricalCrossEntropy(vector<double> target, vector<double> output)
{
	if (target.size() != output.size()) { printf("Cross Entropy Loss Error: �������� �������� �迭 ũ�Ⱑ ��ġ���� ����.\n"); return 0.0; }
	size_t error_length = target.size(); // = output.size()
	double error = 0;
	for (size_t i = 0; i < error_length; i++) error += target[i] * log(output[i]); // ��(target - ln(output))

	return (-1.0) * error / error_length;
}

double bpl::Functions::CategoricalCrossEntropy(double* target, double* output, int n)
{
	return 0.0;
}

double bpl::Functions::SparseCrossEntropyLoss(vector<double> target, vector<double> output)
{
	return 0.0;
}

double bpl::Functions::hingeLoss()
{
	return 0.0;
}