#include "Model.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace bpl;

int main(int argc, char* argv[])
{
	hyper_parameters params;
	params.optimizer = Optimizer::Adam;
	params.weight_init = WeightInitMethod::NormalDistribution;
	params.learning_rate = 0.1;
	early_stopping_parameters early_stopping;
	early_stopping.patience = 10;
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

	model.addLayer(3, ActiveFunction::Sigmoid);
	model.addLayer(1, ActiveFunction::Sigmoid);
	model.prepare(params);

	model.printModel();

	model.learning(6000, 1, early_stopping, verbose);

	model.predictToFileFromFile("input2-test.txt", " ", "output.txt", true);
	model.saveModel();

	return 0;
}
