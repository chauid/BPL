#include "Model.h"
#include <stdio.h>
#include <stdlib.h>

/*
* MSE 미분 시 Sigmoid의 1 - out_p에서 out_p의 값이 극단적으로 1.0에 가까워지면서 0을 출력 => 델타값 = 0
* 출력값 클리핑
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
	params.weight_init = WeightInitMethod::XavierNormalDistribution;
	params.learning_rate = 0.1;
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

	model.addDenseLayer(72, ActiveFunction::Sigmoid);
	model.addDenseLayer(2, ActiveFunction::Sigmoid);
	model.addDenseLayer(1, ActiveFunction::Sigmoid);
	model.prepare(params);

	model.printModel();

	model.learning(1000, 1, early_stopping, verbose);

	model.predictToFileFromFile("input2-test.txt", " ", "output.txt", true);
	//model.predictToFileFromFile("ThoraricSurgery3-test.txt", " ", "output.txt", true);
	model.saveModel();

	return 0;
}
