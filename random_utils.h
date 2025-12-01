#pragma once
#include <random>

namespace random_utils {
	static std::default_random_engine rand_gen(std::random_device{}());

	namespace sample {
		bool bit(double p=0.5) {
			return (std::bernoulli_distribution(p))(rand_gen);
		}

		double uniform(double min = 0.0, double max = 1.0) {
			return (std::uniform_real_distribution<double>(min, max))(rand_gen);
		}

		double normal(double mean = 0.0, double var = 1.0) {
			return (std::normal_distribution<double>(mean, sqrt(var)))(rand_gen);
		}

		double gamma(double shape, double scale) {
			return (std::gamma_distribution<double>(shape, scale))(rand_gen);
		}

		double invgamma(double shape, double scale) {
			return 1.0 / sample::gamma(shape, 1/scale);
		}

		double beta(double a, double b) {
			double x = sample::gamma(a, 1), y = sample::gamma(b, 1);
			return x / (x + y);
		}

		size_t binom(size_t n, double p) {
			return (std::binomial_distribution<size_t>(n, p))(rand_gen);
		}

		size_t index(size_t max_index) {
			return (std::uniform_int_distribution<size_t>(0, max_index - 1))(rand_gen);
		}

		size_t weight_index(const std::vector<double>& weights) {
			return (std::discrete_distribution<size_t>(weights.begin(), weights.end()))(rand_gen);
		}

		std::vector<size_t> multiple_index(size_t max_index, size_t n) {
			std::vector<size_t> sampled, candidates(max_index);
			sampled.reserve(n);
			iota(candidates.begin(), candidates.end(), 0);

			while (n-- > 0) {
				auto smp = index(candidates.size());
				sampled.push_back(candidates[smp]);

				candidates[smp] = candidates.back();
				candidates.pop_back();
			}

			return sampled;
		}

		std::vector<size_t> weight_multiple_index(std::vector<double> weights, size_t n) {
			std::vector<size_t> sampled, candidates(weights.size());
			sampled.reserve(n);
			iota(candidates.begin(), candidates.end(), 0);

			while (n-- > 0) {
				auto smp = weight_index(weights);
				sampled.push_back(candidates[smp]);

				candidates[smp] = candidates.back();
				candidates.pop_back();

				weights[smp] = weights.back();
				weights.pop_back();
			}

			return sampled;
		}

		template<typename ct>
		std::set<ct> random_subset(const std::set<ct>& s, size_t p) {
			const auto n = s.size();
			if (p > n)
				throw std::invalid_argument("p > s.size()");
			std::vector<ct> current;
			auto it = s.begin();
			size_t i = 0;
			for (; i < p; i++) {
				current.push_back(*(it++));
			}
			for (; i < n; i++) {
				size_t j = std::uniform_int_distribution<size_t>(0, i)(rand_gen);
				if (j < p) {
					current[j] = *it;
				}
				++it;
			}
			return std::set<ct>(current.begin(), current.end());
		}

	}
	
	namespace density {
		double normal(double x, double mean=0.0, double var=1.0) {
			const double sig = sqrt(var);
			const auto&& r = x - mean;
			return 0.3989422804 / sig * exp(-r*r / 2 / var);
		}

		double normal_log(double x, double mean=0.0, double var=1.0) {
			return -0.9189385332-(pow(x-mean, 2)/2/var + 0.5 * log(var));
		}
	}
};