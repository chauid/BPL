#include "Model.h"
#include <stdio.h>
#include <stdlib.h>

/*
* 최적화 함수에서 완전한 GPU 연산으로 위임
* OpenCV 사용해서 이미지 픽셀 읽기 및 Flatten함수 구현
*/

using namespace std;
using namespace bpl;

int main(int argc, char* argv[])
{
	//const double epsilon = 0.0000001; // 1e-07
	//double x = 2 * -1 * epsilon * -1;
	//printf("x = %.9lf\n", x);
	//system("pause");
	//double x = Functions::Sigmoid(1.99);
	//cout << "x= " << x << '\n';

	hyper_parameters params;
	params.optimizer = Optimizer::Adam;
	params.loss_function = LossFunction::MSE;
	params.weight_init = WeightInitMethod::XavierNormalDistribution;
	params.learning_rate = 0.001;
	early_stopping_parameters early_stopping;
	early_stopping.patience = 0;
	//early_stopping.min_loss = 0.0001;
	//early_stopping.start_from_epoch = 50;
	learning_verbose_parameters verbose;
	verbose.error_verbose = true;
	//verbose.write_file = true;


	vector<double> predict;

	Model model;
	model.debug_mode = false;
	model.verbose_time = false;

	model.readInputData("input2.txt");

	model.addDenseLayer(300, ActiveFunction::Sigmoid);
	model.addDenseLayer(1, ActiveFunction::Sigmoid);
	model.prepare(params);

	model.printModel();

	model.learning(3000, 2, early_stopping, verbose);
	//model.clearModel();
	//model.loadModel();

	model.predictToFileFromFile("input2-test.txt", " ", "output.txt", true);
	//model.predictToFileFromFile("ThoraricSurgery3-test.txt", " ", "output.txt", true);
	model.saveModel();

	return 0;
}
