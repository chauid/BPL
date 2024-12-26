#include "Model.h"
#include <stdio.h>
#include <stdlib.h>

/*
* TODO List
* bias update 구현
* 최적화 함수에서 완전한 GPU 연산으로 위임: cuda.cu
* OpenCV 사용해서 이미지 픽셀 읽기 및 Flatten함수 구현
*/

using namespace std;
using namespace bpl;

int main(int argc, char* argv[])
{
	hyper_parameters params;
	params.optimizer = Optimizer::Adam;
	params.loss_function = LossFunction::MSE;
	params.weight_init = WeightInitMethod::XavierNormalDistribution;
	params.learning_rate = 0.001;
	early_stopping_parameters early_stopping;
	early_stopping.patience = 50;
	//early_stopping.min_loss = 0.0001;
	//early_stopping.start_from_epoch = 50;
	learning_verbose_parameters verbose;
	verbose.error_verbose = true;
	//verbose.write_file = true;


	//vector<double> predict;

	Model model;
	//model.debug_mode = false;
	//model.verbose_time = false;

	model.readInputData("input2.txt");

	//model.addDenseLayer(6, ActiveFunction::ReLU);
	//model.addDenseLayer(12, ActiveFunction::ReLU);
	//model.addDenseLayer(1, ActiveFunction::Sigmoid);
	//model.prepare(params);
	//model.printModel();


	//model.learning(3000, 1, early_stopping, verbose);
	//model.predictToFileFromFile("input2-test.txt", " ", "output.txt", true);
	//model.predictToFileFromFile("ThoraricSurgery3-test.txt", " ", "output.txt", true);
	//model.saveModel();
	model.loadModel();
	model.printModel();
	vector<vector<double>> result = model.predictFromFile("input2-test.txt");
	for (vector<double> result_i : result)
	{
		for (double result_j : result_i)
		{
			cout << result_j << ' ';
		}
		cout << '\n';
	}


	return 0;
}
