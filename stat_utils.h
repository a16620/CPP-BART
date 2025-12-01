#pragma once
#include "data_matrix.h"
#include <exception>
#include <algorithm>
#include <numeric>
#include <cmath>
using namespace std;

namespace stat_utils {
	double gamma_incomplete_upper(double x, double a, size_t n = 500) {
		if (x <= 0.0 || a <= 0.0)
			return -1;


		const double delta = max(5.0 * a - x, 100.0) / n;
		double loc = x, sum = std::pow(loc, a - 1.0) * std::exp(-loc) * 0.5;
		for (size_t i = 1; i < n; ++i) {
			loc += delta;
			sum += std::pow(loc, a - 1.0) * std::exp(-loc);
		}

		return sum * delta;
	}

	std::pair<double, double> range(const std::vector<double>& x) {
		auto range_it = minmax_element(x.cbegin(), x.cend());
		return make_pair(*range_it.first, *range_it.second);
	}

	double crossprod(const std::vector<double>& x) {
		return accumulate(x.cbegin(), x.cend(), 0.0, [](const double& sum, double xi) {return sum+xi*xi;});
	}

	double crossprod(const std::vector<double>& x, const std::vector<double>& y) {
		if (x.size() != y.size())
			throw runtime_error("x.size() != y.size()");
		double sum = 0.0;
		size_t&& max_i = x.size();
		for (size_t i = 0; i < max_i; i++) {
			sum += x[i] * y[i];
		}
		return sum;
	}

	//X는 정렬되어 있어야 함. q: [0, 1.0].
	double quantile(const std::vector<double>& x, double left_q) {
		const auto n = x.size();
		double percent_n = (n-1)*left_q;
		auto index_0 = static_cast<size_t>(percent_n);
		double inter = percent_n - index_0;
		if (index_0 > (n-1))
			throw invalid_argument("x의 길이가 0이거나 q가 범위를 벗어났습니다");
		else if (index_0 == (n-1))
			return x.back();
		return (x.at(index_0)*(1-inter) + x.at(index_0 + 1)*inter);
	}

	std::pair<double, double> sum2(const std::vector<double>& x) {
		double s = 0, s2 = 0;
		for (const auto& xi : x) {
			s += xi;
			s2 += xi * xi;
		}
		return make_pair(s, s2);
	}

	inline double square(const double& x) {
		return x*x;
	}

	template<typename numeric_type>
	double sum(const std::vector<numeric_type>& x) {
		return accumulate(x.cbegin(), x.cend(), 0.0);
	}

	template<typename numeric_type>
	double mean(const std::vector<numeric_type>& x) {
		return sum(x) / x.size();
	}

	std::pair<double, double> mean_var(const std::vector<double>& x) {
		double sample_mean = stat_utils::mean(x);
		double sample_var = 0.0;
		for (const auto& xx : x) {
			const auto&& r = xx - sample_mean;
			sample_var += r*r;
		}
		return make_pair(sample_mean, sample_var / (x.size() - 1));
	}

	std::pair<double, double> mean_var2(const std::vector<double>& x) {
		double sample_mean = stat_utils::mean(x);
		double sample_var = 0.0;
		for (const auto& xx : x) {
			const auto&& r = xx - sample_mean;
			sample_var += r*r;
		}
		return make_pair(sample_mean, sample_var);
	}

	double LR_var_estimate(DataMatrix X, const std::vector<double>& Y, bool add_intercept=true) {
		if (add_intercept) {
			X.append_numeric(1.0);
		}


		X = X.Hat_by_cholesky();
		DataMatrix residual = (DataMatrix::identity(X.nrow())-X)*Y;
		auto& residual_vec = residual.raw_num_vec();
		auto res = mean_var(residual_vec);
		return res.second;
	}
}