#pragma once
#include <memory>
#include <vector>
#include <random>
#include <ctime>

struct Function {
	std::vector<double> f;
	double xmin;
	double xmax;
	double deltax;
	double offset;
	int index;
};

void InitializeFunction(Function& F, int nPoints,
	double xmin, double xmax, double fmin, double fmax, std::mt19937& rng) {

	F.f.resize(nPoints);
	std::uniform_real_distribution<double> dist(fmin, fmax);
	for (int j = 0; j < nPoints; ++j) {
		F.f[j] = dist(rng);
	}
	F.xmin = xmin;
	F.xmax = xmax;

	double gap = 0.01 * (F.xmax - F.xmin);
	F.xmin -= gap;
	F.xmax += gap;
	F.deltax = (F.xmax - F.xmin) / (nPoints - 1);
}

double Compute(double x, bool freezeModel, Function& F) {
	if (!freezeModel) {
		bool isChanged = false;
		if (x <= F.xmin) {
			F.xmin = x;
			isChanged = true;
		}
		else if (x >= F.xmax) {
			F.xmax = x;
			isChanged = true;
		}
		if (isChanged) {
			double gap = 0.01 * (F.xmax - F.xmin);
			F.xmin -= gap;
			F.xmax += gap;
			F.deltax = (F.xmax - F.xmin) / (F.f.size() - 1);
		}
	}
	if (x <= F.xmin) {
		F.index = 0;
		F.offset = 0.001;
		return F.f[0];
	}
	else if (x >= F.xmax) {
		F.index = (int)(F.f.size()) - 2;
		F.offset = 0.999;
		return F.f[F.f.size() - 1];
	}
	else {
		double R = (x - F.xmin) / F.deltax;
		F.index = (int)(R);
		F.offset = R - F.index;
		return F.f[F.index] + (F.f[F.index + 1] - F.f[F.index]) * F.offset;
	}
}

double ComputeDerivative(Function& F) {
	return (F.f[F.index + 1] - F.f[F.index]) / F.deltax;
}

void Update(double delta, Function& F) {
	double tmp = delta * F.offset;
	F.f[F.index + 1] += tmp;
	F.f[F.index] += delta - tmp;
}
