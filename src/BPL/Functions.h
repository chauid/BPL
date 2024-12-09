#pragma once
#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

namespace bpl
{
	class Functions
	{
	public:
		static void printArray(const char* arr_name, int length, double* arr, int ld_arr);

		/// <summary>
		/// ��� ���� �Լ�: ����ġ ���(m * n), �Է� ���(n * r) | Row-Major
		/// </summary>
		/// <param name="matrix1">�����</param>
		/// <param name="matrix2">�����</param>
		/// <returns>(m * r) ���</returns>
		static vector<vector<double>> matrixMultiplication(vector<vector<double>> matrix1, vector<vector<double>> matrix2);

		static double* matrixMultiplication(int m, int n, int k, double* A, double* B);

		static vector<int> oneHotEncoding(vector<double> data);

		static void oneHotEncoding(double* data, int num_classes);

		/* ------------------------------------------------------------------------------------------------------------------------ */
		/* Active Functions */

		/// <summary>
		/// Sigmoid Function
		/// </summary>
		/// <param name="x"></param>
		/// <returns>1 / (1 + exp(-x))</returns>
		static double Sigmoid(double x);

		/// <summary>
		/// Sigmoid Function
		/// </summary>
		/// <param name="a">x����</param>
		/// <param name="b">y����</param>
		/// <param name="x"></param>
		/// <returns>1 / (1 + exp(-ax + b))</returns>
		static double Sigmoid(double a, double b, double x);

		/// <summary>
		/// Hyperbolic Tangent
		/// </summary>
		/// <param name="x"></param>
		/// <returns>(2 / 1 + exp(-2x)) - 1 = 2 * sigmoid(2 * x) - 1</returns>
		static double tanh(double x);

		/// <summary>
		/// Softmax Function: not original softmax
		/// </summary>
		/// <param name="x"></param>
		/// <returns>Index of each x in list: n | exp(x[n] - x.max) / sum(exp(x[n] - x.max))</returns>
		static vector<double> Softmax(vector<double> x);


		/// <summary>
		/// Softmax Function: not original softmax
		/// </summary>
		/// <param name="x"></param>
		/// <param name="x_length"></param>
		/// <returns>Index of each x in list: n | exp(x[n] - x.max) / sum(exp(x[n] - x.max))</returns>
		static void Softmax(double* x, int x_length);

		/// <summary>
		/// ReLu Function
		/// </summary>
		/// <param name="x"></param>
		/// <returns>x > 0 ? x : 0</returns>
		static double ReLU(double x);

		/// <summary>
		/// Leaky ReLu Function
		/// </summary>
		/// <param name="x"></param>
		/// <returns>x > 0 ? x : 0.01x</returns>
		static double leakyReLU(double x);

		/// <summary>
		/// Exponential Linear Unit
		/// </summary>
		/// <param name="x"></param>
		/// <returns>x > 0 ? x : exp(x) - 1</returns>
		static double ELU(double x);

		/// <summary>
		/// Swish
		/// </summary>
		/// <param name="x"></param>
		/// <param name="beta"></param>
		/// <returns>x * (1 / (1 + exp(-x))) = x * sigmoid(x)</returns>
		static double Swish(double x, double beta = 1.0);

		/* ------------------------------------------------------------------------------------------------------------------------ */
		/* Loss Functions */

		/// <summary>
		/// Mean Squared Error(��� ���� ����)
		/// </summary>
		/// <param name="target">������</param>
		/// <param name="output">������</param>
		/// <returns>1/n * ��(target - output)�� | n = target.size() = output.size()</returns>
		static double MSE(vector<double> target, vector<double> output);

		/// <summary>
		/// Mean Squared Error(��� ���� ����)
		/// </summary>
		/// <param name="target">������</param>
		/// <param name="output">������</param>
		/// <param name="n">�迭 ����: ������ ���� = ������ ����</param>
		/// <returns>1/n * ��(target - output)�� | n = target.size() = output.size()</returns>
		static double MSE(double* target, double* output, int n);

		/// <summary>
		/// Binary Cross Entropy Loss(���� ���� ��Ʈ���� �ս� �Լ�)
		/// </summary>
		/// <param name="target">������(One-Hot Vector)</param>
		/// <param name="output">������(Softmax || Sigmoid)</param>
		/// <returns></returns>
		static double BinaryCrossEntropy(vector<double> target, vector<double> output);

		/// <summary>
		/// Categorical Cross Entropy Loss(������ ���� ��Ʈ���� �ս� �Լ�): ���� �з�
		/// </summary>
		/// <param name="target">������(One-Hot Vector)</param>
		/// <param name="output">������(Softmax)</param>
		/// <returns></returns>
		static double CategoricalCrossEntropy(vector<double> target, vector<double> output);

		/// <summary>
		/// Categorical Cross Entropy Loss(������ ���� ��Ʈ���� �ս� �Լ�): ���� �з�
		/// </summary>
		/// <param name="target">������(One-Hot Vector)</param>
		/// <param name="output">������(Softmax)</param>
		/// <param name="n">�迭 ����: ������ ���� = ������ ����</param>
		/// <returns></returns>
		static double CategoricalCrossEntropy(double* target, double* output, int n);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="target"></param>
		/// <param name="output"></param>
		/// <returns></returns>
		static double SparseCrossEntropyLoss(vector<double> target, vector<double> output);

		static double hingeLoss();
	};
}

#endif // !__FUNCTIONS__

