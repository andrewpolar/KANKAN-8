//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// they are under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://link.springer.com/article/10.1007/s10994-025-06800-6

//Website:
//http://OpenKAN.org

#include <iostream>
#include <cmath>
#include <algorithm>
#include "Helper.h"
#include "Function.h"

void Determinants44() {
	const int nTrainingRecords = 100'000;
	const int nValidationRecords = 20'000;
	const int nMatrixSize = 4;
	const int nFeatures = nMatrixSize * nMatrixSize;
	double min = 0.0;
	double max = 10.0;

	auto features_training = GenerateInput(nTrainingRecords, nFeatures, min, max);
	auto features_validation = GenerateInput(nValidationRecords, nFeatures, min, max);
	auto targets_training = ComputeDeterminantTarget(features_training, nMatrixSize);
	auto targets_validation = ComputeDeterminantTarget(features_validation, nMatrixSize);

	clock_t start_application = clock();
	clock_t current_time = clock();

	double targetMin = *std::min_element(targets_training.begin(), targets_training.end());
	double targetMax = *std::max_element(targets_training.begin(), targets_training.end());

	const int nInner = 70;
	const int nOuter = 1;
	const double alpha = 0.1;
	const int nInnerPoints = 3;
	const int nOuterPoints = 30;
	const double termination = 0.97;

	// Instantiate models
	std::random_device rd;
	std::mt19937 rng(rd());
	std::vector<std::unique_ptr<Function>> innerFunctions;
	for (int i = 0; i < nInner * nFeatures; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nInnerPoints, min, max, targetMin, targetMax, rng);
		innerFunctions.push_back(std::move(function));
	}
	std::vector<std::unique_ptr<Function>> outerFunctions;
	for (int i = 0; i < nInner; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nOuterPoints, targetMin, targetMax, targetMin, targetMax, rng);
		outerFunctions.push_back(std::move(function));
	}

	//auxiliary buffers
	std::vector<double> models0(nInner);
	std::vector<double> models1(nOuter);

	std::vector<double> deltas0(nInner);
	std::vector<double> deltas1(nOuter);
	std::vector<double> predictions(nValidationRecords);

	//training
	printf("Targets are determinants of random 4 * 4 matrices, %d training records\n", nTrainingRecords);
	for (int epoch = 0; epoch < 32; ++epoch) {
		for (int record = 0; record < nTrainingRecords; ++record) {
			for (int k = 0; k < nInner; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_training[record][j], false, *innerFunctions[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}

			for (int k = 0; k < nOuter; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nInner; ++j) {
					models1[k] += Compute(models0[j], false, *outerFunctions[j]);
				}
				models1[k] /= nInner;
			}

			//in general, for vector target, deltas are vectors and derivatives 
			//are matrices, and computing of next deltas is matrix vector 
			//multiplication, but this is particular case with scalar target
			deltas1[0] = alpha * (targets_training[record] - models1[0]);
			for (int j = 0; j < nInner; ++j) {
				deltas0[j] = deltas1[0] * ComputeDerivative(*outerFunctions[j]);
			}

			for (int k = 0; k < nOuter; ++k) {
				for (int j = 0; j < nInner; ++j) {
					Update(deltas1[k], *outerFunctions[j]);
				}
			}

			for (int k = 0; k < nInner; ++k) {
				for (int j = 0; j < nFeatures; ++j) {
					Update(deltas0[k], *innerFunctions[k * nFeatures + j]);
				}
			}
		}

		//validation
		for (int record = 0; record < nValidationRecords; ++record) {
			for (int k = 0; k < nInner; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_validation[record][j], true, *innerFunctions[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nOuter; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nInner; ++j) {
					models1[k] += Compute(models0[j], true, *outerFunctions[j]);
				}
				models1[k] /= nInner;
			}
			predictions[record] = models1[0];
		}
		double pearson = Pearson(predictions, targets_validation);

		current_time = clock();
		printf("%d pearson %4.3f, Time %2.3f\n", epoch, pearson, (double)(current_time - start_application) / CLOCKS_PER_SEC);
		if (pearson >= termination) break;
	}
	printf("\n");
}

void AreasOfTriangles() {
	//parameters of dataset
	const int nFeatures = 6;
	const int nTrainingRecords = 10'000;
	const int nValidationRecords = 2'000;
	const double min = 0.0;
	const double max = 1.0;

	//making datasets
	auto features_training = MakeRandomMatrixForTriangles(nTrainingRecords, nFeatures, min, max);
	auto features_validation = MakeRandomMatrixForTriangles(nValidationRecords, nFeatures, min, max);
	auto targets_training = ComputeAreasOfTriangles(features_training);
	auto targets_validation = ComputeAreasOfTriangles(features_validation);

	//data is ready, we start training
	clock_t start_application = clock();
	clock_t current_time = clock();

	//find limits for targets
	double targetMin = *std::min_element(targets_training.begin(), targets_training.end());
	double targetMax = *std::max_element(targets_training.begin(), targets_training.end());

	//configuration of network
	const int nU0 = 50;
	const int nU1 = 10;
	const int nU2 = 4;
	const int nU3 = 1;
	const int nPoints0 = 2;
	const int nPoints1 = 12;
	const int nPoints2 = 12;
	const int nPoints3 = 22;
	const double alpha = 0.005;
	const int nEpochs = 32;
	const double termination = 0.99;

	// Instantiate models
	std::random_device rd;
	std::mt19937 rng(rd());

	std::vector<std::unique_ptr<Function>> layer0;
	for (int i = 0; i < nU0 * nFeatures; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints0, min, max, targetMin, targetMax, rng);
		layer0.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer1;
	for (int i = 0; i < nU1 * nU0; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints1, targetMin, targetMax, targetMin, targetMax, rng);
		layer1.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer2;
	for (int i = 0; i < nU2 * nU1; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints2, targetMin, targetMax, targetMin, targetMax, rng);
		layer2.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer3;
	for (int i = 0; i < nU3 * nU2; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints3, targetMin, targetMax, targetMin, targetMax, rng);
		layer3.push_back(std::move(function));
	}

	//auxiliary buffers
	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> models2(nU2);
	std::vector<double> models3(nU3);

	std::vector<std::vector<double>> derivatives2(nU3, std::vector<double>(nU2));
	std::vector<std::vector<double>> derivatives1(nU2, std::vector<double>(nU1));
	std::vector<std::vector<double>> derivatives0(nU1, std::vector<double>(nU0));

	std::vector<double> deltas3(nU3);
	std::vector<double> deltas2(nU2);
	std::vector<double> deltas1(nU1);
	std::vector<double> deltas0(nU0);

	auto v = std::vector<double>(nValidationRecords);

	printf("Targets are areas of random triangles, %d training records\n", nTrainingRecords);
	for (int epoch = 0; epoch < nEpochs; ++epoch) {
		//training
		for (int record = 0; record < nTrainingRecords; ++record) {
			//steps: forward pass layer by layer
			for (int k = 0; k < nU0; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_training[record][j], false, *layer0[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nU1; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nU0; ++j) {
					models1[k] += Compute(models0[j], false, *layer1[k * nU0 + j]);
				}
				models1[k] /= nU0;
			}
			for (int k = 0; k < nU2; ++k) {
				models2[k] = 0.0;
				for (int j = 0; j < nU1; ++j) {
					models2[k] += Compute(models1[j], false, *layer2[k * nU1 + j]);
				}
				models2[k] /= nU1;
			}
			for (int k = 0; k < nU3; ++k) {
				models3[k] = 0.0;
				for (int j = 0; j < nU2; ++j) {
					models3[k] += Compute(models2[j], false, *layer3[k * nU2 + j]);
				}
				models3[k] /= nU2;
			}
			//end of forward pass

			//compute all derivative matrices
			for (int k = 0; k < nU3; ++k) {
				for (int j = 0; j < nU2; ++j) {
					derivatives2[k][j] = ComputeDerivative(*layer3[k * nU2 + j]);
				}
			}
			for (int k = 0; k < nU2; ++k) {
				for (int j = 0; j < nU1; ++j) {
					derivatives1[k][j] = ComputeDerivative(*layer2[k * nU1 + j]);
				}
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					derivatives0[k][j] = ComputeDerivative(*layer1[k * nU0 + j]);
				}
			}

			//compute all delta vectors for each layer
			deltas3[0] = alpha * (targets_training[record] - models3[0]);  //firs one is scalar

			for (int j = 0; j < nU2; ++j) {
				deltas2[j] = 0.0;
				for (int i = 0; i < nU3; ++i) {
					deltas2[j] += derivatives2[i][j] * deltas3[i];
				}
			}
			for (int j = 0; j < nU1; ++j) {
				deltas1[j] = 0.0;
				for (int i = 0; i < nU2; ++i) {
					deltas1[j] += derivatives1[i][j] * deltas2[i];
				}
			}
			for (int j = 0; j < nU0; ++j) {
				deltas0[j] = 0.0;
				for (int i = 0; i < nU1; ++i) {
					deltas0[j] += derivatives0[i][j] * deltas1[i];
				}
			}

			//step: update all layers by deltas
			for (int k = 0; k < nU3; ++k) {
				for (int j = 0; j < nU2; ++j) {
					Update(deltas3[k], *layer3[k * nU2 + j]);
				}
			}
			for (int k = 0; k < nU2; ++k) {
				for (int j = 0; j < nU1; ++j) {
					Update(deltas2[k], *layer2[k * nU1 + j]);
				}
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					Update(deltas1[k], *layer1[k * nU0 + j]);
				}
			}
			for (int k = 0; k < nU0; ++k) {
				for (int j = 0; j < nFeatures; ++j) {
					Update(deltas0[k], *layer0[k * nFeatures + j]);
				}
			}
		}

		//validation
		for (int record = 0; record < nValidationRecords; ++record) {
			for (int k = 0; k < nU0; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_validation[record][j], true, *layer0[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nU1; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nU0; ++j) {
					models1[k] += Compute(models0[j], true, *layer1[k * nU0 + j]);
				}
				models1[k] /= nU0;
			}
			for (int k = 0; k < nU2; ++k) {
				models2[k] = 0.0;
				for (int j = 0; j < nU1; ++j) {
					models2[k] += Compute(models1[j], true, *layer2[k * nU1 + j]);
				}
				models2[k] /= nU1;
			}
			for (int k = 0; k < nU3; ++k) {
				models3[k] = 0.0;
				for (int j = 0; j < nU2; ++j) {
					models3[k] += Compute(models2[j], true, *layer3[k * nU2 + j]);
				}
				models3[k] /= nU2;
			}
			v[record] = models3[0];
		}

		double pearson = Pearson(v, targets_validation);

		current_time = clock();
		printf("Epoch %d, Pearson: %f, time %2.3f\n", epoch, pearson,
			(double)(current_time - start_application) / CLOCKS_PER_SEC);
		if (pearson >= termination) break;
	}
	printf("\n");
}

void Medians() {
	//data
	const int nTrainingRecords = 10'000;
	const int nValidationRecords = 2'000;
	const int nFeatures = 6;
	const int nTargets = 3;
	const double min = 0.0;
	const double max = 1.0;

	//data generation
	auto features_training = GenerateInputsMedians(nTrainingRecords, nFeatures, min, max);
	auto features_validation = GenerateInputsMedians(nValidationRecords, nFeatures, min, max);
	auto targets_training = ComputeTargetsMedians(features_training);
	auto targets_validation = ComputeTargetsMedians(features_validation);

	//data is ready, we start training
	clock_t start_application = clock();
	clock_t current_time = clock();

	double targetMin = targets_training[0][0];
	double targetMax = targets_training[0][0];
	for (int i = 0; i < nTrainingRecords; ++i) {
		for (int j = 0; j < nTargets; ++j) {
			if (targets_training[i][j] < targetMin) targetMin = targets_training[i][j];
			if (targets_training[i][j] > targetMax) targetMax = targets_training[i][j];
		}
	}

	//network configuration
	const int nU0 = 20;
	const int nU1 = 10;
	const int nU2 = 4;
	const int nU3 = nTargets;
	const int nPoints0 = 2;
	const int nPoints1 = 12;
	const int nPoints2 = 12;
	const int nPoints3 = 22;
	const double alpha = 0.005;
	const int nEpochs = 64;
	const double termination = 0.985;

	//Instantiate models
	std::random_device rd;
	std::mt19937 rng(rd());

	std::vector<std::unique_ptr<Function>> layer0;
	for (int i = 0; i < nU0 * nFeatures; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints0, min, max, targetMin, targetMax, rng);
		layer0.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer1;
	for (int i = 0; i < nU1 * nU0; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints1, targetMin, targetMax, targetMin, targetMax, rng);
		layer1.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer2;
	for (int i = 0; i < nU2 * nU1; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints2, targetMin, targetMax, targetMin, targetMax, rng);
		layer2.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer3;
	for (int i = 0; i < nU3 * nU2; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints3, targetMin, targetMax, targetMin, targetMax, rng);
		layer3.push_back(std::move(function));
	}

	//auxiliary buffers
	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> models2(nU2);
	std::vector<double> models3(nU3);

	std::vector<std::vector<double>> derivatives2(nU3, std::vector<double>(nU2));
	std::vector<std::vector<double>> derivatives1(nU2, std::vector<double>(nU1));
	std::vector<std::vector<double>> derivatives0(nU1, std::vector<double>(nU0));

	std::vector<double> deltas3(nU3);
	std::vector<double> deltas2(nU2);
	std::vector<double> deltas1(nU1);
	std::vector<double> deltas0(nU0);

	auto actual0 = std::vector<double>(nValidationRecords);
	auto actual1 = std::vector<double>(nValidationRecords);
	auto actual2 = std::vector<double>(nValidationRecords);

	auto computed0 = std::vector<double>(nValidationRecords);
	auto computed1 = std::vector<double>(nValidationRecords);
	auto computed2 = std::vector<double>(nValidationRecords);

	printf("Targets are medians of random triangles, %d training records\n", nTrainingRecords);
	for (int epoch = 0; epoch < nEpochs; ++epoch) {
		//training
		for (int record = 0; record < nTrainingRecords; ++record) {
			//steps: forward pass layer by layer
			for (int k = 0; k < nU0; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_training[record][j], false, *layer0[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nU1; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nU0; ++j) {
					models1[k] += Compute(models0[j], false, *layer1[k * nU0 + j]);
				}
				models1[k] /= nU0;
			}
			for (int k = 0; k < nU2; ++k) {
				models2[k] = 0.0;
				for (int j = 0; j < nU1; ++j) {
					models2[k] += Compute(models1[j], false, *layer2[k * nU1 + j]);
				}
				models2[k] /= nU1;
			}
			for (int k = 0; k < nU3; ++k) {
				models3[k] = 0.0;
				for (int j = 0; j < nU2; ++j) {
					models3[k] += Compute(models2[j], false, *layer3[k * nU2 + j]);
				}
				models3[k] /= nU2;
			}
			//end of forward pass

			//compute all derivative matrices
			for (int k = 0; k < nU3; ++k) {
				for (int j = 0; j < nU2; ++j) {
					derivatives2[k][j] = ComputeDerivative(*layer3[k * nU2 + j]);
				}
			}
			for (int k = 0; k < nU2; ++k) {
				for (int j = 0; j < nU1; ++j) {
					derivatives1[k][j] = ComputeDerivative(*layer2[k * nU1 + j]);
				}
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					derivatives0[k][j] = ComputeDerivative(*layer1[k * nU0 + j]);
				}
			}

			//compute all deltas for all updates
			for (int j = 0; j < nU3; ++j) {
				deltas3[j] = (targets_training[record][j] - models3[j]) * alpha;
			}

			for (int j = 0; j < nU2; ++j) {
				deltas2[j] = 0.0;
				for (int i = 0; i < nU3; ++i) {
					deltas2[j] += derivatives2[i][j] * deltas3[i];
				}
			}
			for (int j = 0; j < nU1; ++j) {
				deltas1[j] = 0.0;
				for (int i = 0; i < nU2; ++i) {
					deltas1[j] += derivatives1[i][j] * deltas2[i];
				}
			}
			for (int j = 0; j < nU0; ++j) {
				deltas0[j] = 0.0;
				for (int i = 0; i < nU1; ++i) {
					deltas0[j] += derivatives0[i][j] * deltas1[i];
				}
			}

			//step: update all layers by deltas
			for (int k = 0; k < nU3; ++k) {
				for (int j = 0; j < nU2; ++j) {
					Update(deltas3[k], *layer3[k * nU2 + j]);
				}
			}
			for (int k = 0; k < nU2; ++k) {
				for (int j = 0; j < nU1; ++j) {
					Update(deltas2[k], *layer2[k * nU1 + j]);
				}
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					Update(deltas1[k], *layer1[k * nU0 + j]);
				}
			}
			for (int k = 0; k < nU0; ++k) {
				for (int j = 0; j < nFeatures; ++j) {
					Update(deltas0[k], *layer0[k * nFeatures + j]);
				}
			}
		}

		for (int record = 0; record < nValidationRecords; ++record) {
			for (int k = 0; k < nU0; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_training[record][j], true, *layer0[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nU1; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nU0; ++j) {
					models1[k] += Compute(models0[j], true, *layer1[k * nU0 + j]);
				}
				models1[k] /= nU0;
			}
			for (int k = 0; k < nU2; ++k) {
				models2[k] = 0.0;
				for (int j = 0; j < nU1; ++j) {
					models2[k] += Compute(models1[j], true, *layer2[k * nU1 + j]);
				}
				models2[k] /= nU1;
			}
			for (int k = 0; k < nU3; ++k) {
				models3[k] = 0.0;
				for (int j = 0; j < nU2; ++j) {
					models3[k] += Compute(models2[j], true, *layer3[k * nU2 + j]);
				}
				models3[k] /= nU2;
			}

			actual0[record] = targets_validation[record][0];
			actual1[record] = targets_validation[record][1];
			actual2[record] = targets_validation[record][2];

			computed0[record] = models3[0];
			computed1[record] = models3[1];
			computed2[record] = models3[2];
		}

		//pearsons for correlated targets
		double p1 = Pearson(computed0, actual0);
		double p2 = Pearson(computed1, actual1);
		double p3 = Pearson(computed2, actual2);

		current_time = clock();
		printf("Epoch %d, Pearsons for validation: %f, %f, %f, time %2.3f\n", epoch, p1, p2, p3,
			(double)(current_time - start_application) / CLOCKS_PER_SEC);

		if (p1 > termination && p2 > termination && p3 > termination) break;
	}
	printf("\n");
}

void Tetrahedrons() {
	//data
	const int nTrainingRecords = 500'000;
	const int nValidationRecords = 50'000;
	const int nFeatures = 12;
	const int nTargets = 4;
	const double min = 0.0;
	const double max = 10.0;

	//generation
	auto features_training = MakeRandomMatrix(nTrainingRecords, nFeatures, min, max);
	auto features_validation = MakeRandomMatrix(nValidationRecords, nFeatures, min, max);
	auto targets_training = ComputeTargetMatrix(features_training);
	auto targets_validation = ComputeTargetMatrix(features_validation);

	//data is ready, we start training
	clock_t start_application = clock();
	clock_t current_time = clock();

	double targetMin = targets_training[0][0];
	double targetMax = targets_training[0][0];
	for (int i = 0; i < nTrainingRecords; ++i) {
		for (int j = 0; j < nTargets; ++j) {
			if (targets_training[i][j] < targetMin) targetMin = targets_training[i][j];
			if (targets_training[i][j] > targetMax) targetMax = targets_training[i][j];
		}
	}

	const int nU0 = 60;
	const int nU1 = 10;
	const int nU2 = nTargets;
	const double alpha = 0.05;
	const int nPoints0 = 2;
	const int nPoints1 = 12;
	const int nPoints2 = 22;
	const int nEpochs = 64;
	const double termination = 0.975;

	//Instantiate models
	std::random_device rd;
	std::mt19937 rng(rd());

	std::vector<std::unique_ptr<Function>> layer0;
	for (int i = 0; i < nU0 * nFeatures; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints0, min, max, targetMin, targetMax, rng);
		layer0.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer1;
	for (int i = 0; i < nU1 * nU0; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints1, targetMin, targetMax, targetMin, targetMax, rng);
		layer1.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer2;
	for (int i = 0; i < nU2 * nU1; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints2, targetMin, targetMax, targetMin, targetMax, rng);
		layer2.push_back(std::move(function));
	}

	//auxiliary buffers
	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> models2(nU2);

	std::vector<std::vector<double>> derivatives1(nU2, std::vector<double>(nU1));
	std::vector<std::vector<double>> derivatives0(nU1, std::vector<double>(nU0));

	std::vector<double> deltas2(nU2);
	std::vector<double> deltas1(nU1);
	std::vector<double> deltas0(nU0);

	auto actual0 = std::vector<double>(nValidationRecords);
	auto actual1 = std::vector<double>(nValidationRecords);
	auto actual2 = std::vector<double>(nValidationRecords);
	auto actual3 = std::vector<double>(nValidationRecords);

	auto computed0 = std::vector<double>(nValidationRecords);
	auto computed1 = std::vector<double>(nValidationRecords);
	auto computed2 = std::vector<double>(nValidationRecords);
	auto computed3 = std::vector<double>(nValidationRecords);

	printf("Targets are areas of faces of random tetrahedrons, %d\n", nTrainingRecords);
	for (int epoch = 0; epoch < nEpochs; ++epoch) {
		//training
		for (int record = 0; record < nTrainingRecords; ++record) {
			//steps: forward pass layer by layer
			for (int k = 0; k < nU0; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_training[record][j], false, *layer0[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nU1; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nU0; ++j) {
					models1[k] += Compute(models0[j], false, *layer1[k * nU0 + j]);
				}
				models1[k] /= nU0;
			}
			for (int k = 0; k < nU2; ++k) {
				models2[k] = 0.0;
				for (int j = 0; j < nU1; ++j) {
					models2[k] += Compute(models1[j], false, *layer2[k * nU1 + j]);
				}
				models2[k] /= nU1;
			}

			//compute all derivative matrices
			for (int k = 0; k < nU2; ++k) {
				for (int j = 0; j < nU1; ++j) {
					derivatives1[k][j] = ComputeDerivative(*layer2[k * nU1 + j]);
				}
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					derivatives0[k][j] = ComputeDerivative(*layer1[k * nU0 + j]);
				}
			}

			//compute deltas
			for (int j = 0; j < nU2; ++j) {
				deltas2[j] = (targets_training[record][j] - models2[j]) * alpha;
			}

			for (int j = 0; j < nU1; ++j) {
				deltas1[j] = 0.0;
				for (int i = 0; i < nU2; ++i) {
					deltas1[j] += derivatives1[i][j] * deltas2[i];
				}
			}
			for (int j = 0; j < nU0; ++j) {
				deltas0[j] = 0.0;
				for (int i = 0; i < nU1; ++i) {
					deltas0[j] += derivatives0[i][j] * deltas1[i];
				}
			}

			//step: update all layers
			for (int k = 0; k < nU2; ++k) {
				for (int j = 0; j < nU1; ++j) {
					Update(deltas2[k], *layer2[k * nU1 + j]);
				}
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					Update(deltas1[k], *layer1[k * nU0 + j]);
				}
			}
			for (int k = 0; k < nU0; ++k) {
				for (int j = 0; j < nFeatures; ++j) {
					Update(deltas0[k], *layer0[k * nFeatures + j]);
				}
			}
		}

		//validation
		for (int record = 0; record < nValidationRecords; ++record) {
			for (int k = 0; k < nU0; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_training[record][j], true, *layer0[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nU1; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nU0; ++j) {
					models1[k] += Compute(models0[j], true, *layer1[k * nU0 + j]);
				}
				models1[k] /= nU0;
			}
			for (int k = 0; k < nU2; ++k) {
				models2[k] = 0.0;
				for (int j = 0; j < nU1; ++j) {
					models2[k] += Compute(models1[j], true, *layer2[k * nU1 + j]);
				}
				models2[k] /= nU1;
			}

			actual0[record] = targets_validation[record][0];
			actual1[record] = targets_validation[record][1];
			actual2[record] = targets_validation[record][2];
			actual3[record] = targets_validation[record][3];

			computed0[record] = models2[0];
			computed1[record] = models2[1];
			computed2[record] = models2[2];
			computed3[record] = models2[3];
		}
		double p1 = Pearson(computed0, actual0);
		double p2 = Pearson(computed1, actual1);
		double p3 = Pearson(computed2, actual2);
		double p4 = Pearson(computed3, actual3);

		current_time = clock();
		printf("Epoch %d, Pearsons: %f %f %f %f, time %2.3f\n", epoch, p1, p2, p3, p4,
			(double)(current_time - start_application) / CLOCKS_PER_SEC);

		if (p1 > termination && p2 > termination && p3 > termination && p4 > termination) break;
	}
	printf("\n");
}

int main() {
	Determinants44();
	AreasOfTriangles();
	Medians();
	Tetrahedrons();
}
