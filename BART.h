#pragma once
#include "BART_base.h"
#include "stat_utils.h"
using namespace std;
#include <iostream>

class BART : public BART_base<double>, public BART_predictable_numeric<double> {
	vector<double> vec_Y, vec_residual_total;
	vector<vector<double>> partial_esimate_Y;
	double transform_y_shift, transform_y_range;

	double prior_node_mean, prior_node_var, prior_sigma_1, prior_sigma_2;
	double mcmc_current_var;

	vector<double> mcmc_var_chain;

	size_t obs_cursor;
		void set_obs_cursor(size_t tree_idx) {
		if (tree_idx >= trees.size())
			throw invalid_argument("트리 개수를 넘는 위치를 접근했습니다");
		obs_cursor = tree_idx;
	}
	
	vector<double> extract_y_obs(const vector<size_t>& index) const {
		const auto& current_obs_pred = partial_esimate_Y[obs_cursor];
		vector<double> out;
		out.reserve(index.size());
		transform(index.cbegin(), index.cend(), back_inserter(out), [&](const size_t& idx) {
			return vec_residual_total.at(idx) + current_obs_pred.at(idx);
			});
		return out;
	}

	double sample_node_pred(const ptree_node& node) override {
		const auto node_obs = extract_y_obs(node->obs_index); //단축 가능 node_obs.size() -> node->obs_index.size(), obs_sum -> accumulate로 transform 없이 바로 합산
		const auto obs_sum = stat_utils::sum(node_obs);

		const double post_var = 1 / (node_obs.size() / mcmc_current_var + 1 / prior_node_var),
			post_mean = post_var * (obs_sum / mcmc_current_var + prior_node_mean / prior_node_var);

		return random_utils::sample::normal(post_mean, post_var);
	}

	double node_likelihood(const ptree_node& node) const override {
		const auto node_obs = extract_y_obs(node->obs_index);
		const size_t n_obs = node_obs.size();
		const auto mv = stat_utils::mean_var2(node_obs);
		const auto mar_1 = mcmc_current_var + n_obs * prior_node_var;

		return log(mcmc_current_var) - 0.5*(log(mar_1) + mv.second*(n_obs-1)/mcmc_current_var + n_obs*stat_utils::square(mv.first-prior_node_mean)/mar_1);
	}

	double sample_sigma() const {
		const double par1 = prior_sigma_1 + vec_residual_total.size() * 0.5,
			par2 = prior_sigma_2 + stat_utils::crossprod(vec_residual_total) * 0.5;

		return random_utils::sample::invgamma(par1, par2);
	}

	void propagate_patch(const mcmc_patch& patch) {
		auto& partial = partial_esimate_Y.at(obs_cursor);
		for (const auto& pt : patch) {
			const double delta = pt.after - pt.before;
			for (const auto& aff_idx : pt.affected) {
				partial.at(aff_idx) = pt.after;
				vec_residual_total.at(aff_idx) -= delta;
			}
		}
	}

	void resample_tree(const ptree& tree) {
		auto& py = partial_esimate_Y[obs_cursor];
		auto leaves = tree->tree_leaves();
		
		for (const auto& leaf : leaves) {
			const auto pred = sample_node_pred(leaf), opred = leaf->prediction, delta = opred - pred;
			leaf->prediction = pred;

			for (const auto& idx : leaf->obs_index) {
				vec_residual_total[idx] += delta;
				py[idx] = pred;
			}
		}
	}

public:
	BART(const DataMatrix& X, const vector<double>& Y, size_t n_trees, double k = 2, double sig_nu = 3, double sig_q = 0.9, double ts_a = 0.95, double ts_b = 1) : BART_base(X, n_trees, ts_a, ts_b), BART_predictable_numeric(n_trees), vec_Y(Y) {
		if (get_predictor_mat().nrow() != vec_Y.size()) {
			throw invalid_argument("X와 Y의 길이가 다릅니다");
		}

		//Y범위 조정
		auto y_range = stat_utils::range(vec_Y);
		transform_y_shift = y_range.first;
		transform_y_range = y_range.second - y_range.first;

		transform(vec_Y.begin(), vec_Y.end(), vec_Y.begin(), [=](double y) {
			return (y - transform_y_shift) / transform_y_range - 0.5;
			});
		
	
		//Y의 분산 추정
		{
			const double nu2 = sig_nu / 2;
			const double sigma_est = stat_utils::LR_var_estimate(X, vec_Y), target = sig_q*tgamma(nu2);

			double chi2_1q = sig_nu;
			for (int i = 0; i < 3000; i++) {
				auto grad = (stat_utils::gamma_incomplete_upper(0.5*chi2_1q, nu2)-target)/(pow(0.5*chi2_1q, nu2-1)*exp(0.5*chi2_1q))*2.0;
				chi2_1q += grad;
				chi2_1q = max(chi2_1q, 1e-3);
			}

			//lambda*nu/Chi2(nu, 1-q) = sigma_est -> IG(nu/2, lambda*nu/2)
			prior_sigma_1 = nu2;
			prior_sigma_2 = sigma_est*chi2_1q * 0.5;

			mcmc_current_var = random_utils::sample::invgamma(prior_sigma_1, prior_sigma_2);
		}

		const auto Y_mv = stat_utils::mean_var(vec_Y);
		prior_node_mean = Y_mv.first / n_trees;
		prior_node_var = Y_mv.second / (k*k) / n_trees;

		const double total_pred_Y = Y_mv.first;
		vec_residual_total.reserve(vec_Y.size());
		transform(vec_Y.begin(), vec_Y.end(), back_inserter(vec_residual_total), [total_pred_Y](const double& yi) {
			return yi - total_pred_Y;
			});

		partial_esimate_Y.assign(n_trees, vector<double>(vec_Y.size(), prior_node_mean));

		for (const auto& tree : trees) {
			tree->get_root()->prediction = prior_node_mean;
		}
	}

	void mcmc(const size_t burn_in, const size_t iteration, const size_t thin=1) {
		const size_t n_tree = trees.size();
		//Burn-in step
		for (size_t it = 1; it <= burn_in; it++) {
			for (size_t tree_idx = 0; tree_idx < n_tree; tree_idx++) {
				set_obs_cursor(tree_idx);
				const auto& tree = trees[tree_idx];
				auto mcmc_result = MCMC_tree_move(tree);
				if (mcmc_result.first) {
					//Update Y
					propagate_patch(mcmc_result.second);
				} else if (random_utils::sample::bit()) {
					resample_tree(tree);
				}
			}
			mcmc_current_var = sample_sigma();
		}

		mcmc_var_chain.assign(iteration, 0.0);

		auto store = get_store();
		store->load_temporal(trees);
		store->flush();

		//vector<bool> changed(n_tree, false);

		//Storing step
		for (size_t it = 1; it <= iteration; it++) {
			for (size_t thin_it = 0; thin_it < thin; thin_it++) {
				//Sample tree structure and node preditctions
				for (size_t tree_idx = 0; tree_idx < n_tree; tree_idx++) {
					set_obs_cursor(tree_idx);
					const auto& tree = trees[tree_idx];
					auto mcmc_result = MCMC_tree_move(tree);
					if (mcmc_result.first) {
						//Update Y
						propagate_patch(mcmc_result.second);
						//changed[tree_idx] = true;
					} else if (random_utils::sample::bit()) {
						resample_tree(tree);
					}
				}

				//Sample sigma
				mcmc_current_var = sample_sigma();
			}
			//Tree 저장
			for (size_t tree_idx = 0; tree_idx < n_tree; tree_idx++) {
				store->load_temporal(trees[tree_idx]);
				/*
				if (changed[tree_idx])
					store->load_temporal(trees[tree_idx]);
				else
					store->repeat_temporal(tree_idx);
				*/
			}
			store->flush();
			//changed.assign(n_tree, false);

			mcmc_var_chain[it-1] = mcmc_current_var;
		}
	}

	void simple_summary() const {
		size_t max_depth = 0;
		for (size_t tree_idx = 0; tree_idx < trees.size(); tree_idx++) {
			const auto& tree = trees[tree_idx];
			max_depth = max(tree->max_depth(), max_depth);
		}
		cout << "max_dpth= " << max_depth << endl;
		cout << "mean siq: " << stat_utils::mean(mcmc_var_chain)*transform_y_range*transform_y_range << endl;
		cout << "mean res: " << stat_utils::mean_var(vec_residual_total).first << ", " << stat_utils::mean_var(vec_residual_total).second << endl;
		cout << " mean: " << prior_node_mean << " var: " << prior_node_var << "\nsig1: " << prior_sigma_1 << " sig2: " << prior_sigma_2 << endl;
	}

	BART_predict predict(const DataMatrix& mat_X, double ci_interval=0.95) const override {
		auto preds = BART_predictable_numeric<double>::predict(mat_X, ci_interval);
		//return preds;
		transform(preds.ci_low.begin(), preds.ci_low.end(), preds.ci_low.begin(), [this](const double& y) {
			return (y + 0.5) * transform_y_range + transform_y_shift;
		});
		transform(preds.ci_up.begin(), preds.ci_up.end(), preds.ci_up.begin(), [this](const double& y) {
			return (y + 0.5) * transform_y_range + transform_y_shift;
		});
		transform(preds.post_mean.begin(), preds.post_mean.end(), preds.post_mean.begin(), [this](const double& y) {
			return (y + 0.5) * transform_y_range + transform_y_shift;
		});
		return preds;
	}
};