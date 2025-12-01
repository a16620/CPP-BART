#pragma once
#include <array>
#include <memory>
#include "CTree.h"
#include "CTree_store.h"
#include "data_matrix.h"
#include <iostream>
using namespace std;

template<typename typeTreeVal>
class BART_base {
	class MCMC_result;
	using pmcmc_result = unique_ptr<MCMC_result>;
	using move_result = pmcmc_result;
	
protected:
	using tree_structure_type = CTree<typeTreeVal>;
	using ptree = shared_ptr<tree_structure_type>;
	using ptree_node = shared_ptr<typename tree_structure_type::tree_node>;
	using wptree_node = weak_ptr<typename tree_structure_type::tree_node>;
	using split_rule = typename tree_structure_type::split_rule;

	vector<ptree> trees;
	DataMatrix mat_X;

	struct pred_transaction {
		double before, after;
		vector<size_t> affected;
	};
	using mcmc_patch = vector<pred_transaction>;

private:
	//tree structure prior
	double prior_alpha, prior_beta;

	enum MOVE {
		MOVE_GROW,
		MOVE_PRUNE,
		MOVE_CHANGE,
		MOVE_END_NUM
	};

	class MCMC_result {
	protected:
		weak_ptr<tree_structure_type> source_tree;
		wptree_node source;

	public:
		double transition_lik;

		MCMC_result() = delete;
		MCMC_result(const MCMC_result&) = delete;

		MCMC_result(const ptree& tree, const ptree_node& node) : source_tree(tree), source(node) {
		}
		virtual ~MCMC_result()=default;

		virtual void Undo() = 0;
		virtual mcmc_patch patch() const = 0;
	};

	class MCMC_Grow : public MCMC_result {
		double pred;
	public:
		MCMC_Grow(const ptree& tree, const ptree_node& node) : MCMC_result(tree, node) {
			if (!node->isLeaf())
				throw runtime_error("Grow되기 전에 백업을 해야합니다");
			pred = node->prediction;
		}

		void Undo() override {
			
			auto tree = this->source_tree.lock();
			auto node = this->source.lock();

			tree->prune_node(node);
			node->prediction = pred;
		}

		mcmc_patch patch() const override {
			auto parent = this->source.lock(), left = parent->left.lock(), right = parent->right.lock();
			return {
				pred_transaction{pred, left->prediction, left->obs_index},
				pred_transaction{pred, right->prediction, right->obs_index}
			};
		}
	};

	class MCMC_Prune : public MCMC_result {
		double pred_left, pred_right;
		vector<size_t> obs_left, obs_right;
		split_rule rule;
	public:
		MCMC_Prune(const ptree& tree, const ptree_node& node) : MCMC_result(tree, node), rule(node->rule) {
			auto left = node->left.lock(), right = node->right.lock();
			if (!left || !right)
				throw runtime_error("Prune되기 전에 백업을 해야합니다");

			pred_left = left->prediction;
			obs_left = left->obs_index;

			pred_right = right->prediction;
			obs_right = right->obs_index;
		}

		void Undo() override {
			auto tree = this->source_tree.lock();
			auto node = this->source.lock();

			tree->split_node(node, rule);
			node->left.lock()->prediction = pred_left;
			node->right.lock()->prediction = pred_right;
		}

		mcmc_patch patch() const override {
			auto parent = this->source.lock();
			return {
				pred_transaction{pred_left, parent->prediction, obs_left},
				pred_transaction{pred_right, parent->prediction, obs_right}
			};
		}

		const split_rule& get_rule() const {
			return rule;
		}
	};

	class MCMC_Change : public MCMC_result {
		vector<double> leaves_pred;
		vector<wptree_node> leaves_ptr;
		split_rule rule;
	public:
		MCMC_Change(const ptree& tree, const ptree_node& node) : MCMC_result(tree, node) {
			MCMC_Change(tree, node, tree->leaves_of_node(node));
		}

		MCMC_Change(const ptree& tree, const ptree_node& node, const vector<ptree_node>& leaves) : MCMC_result(tree, node), rule(node->rule) {
			leaves_pred.reserve(leaves.size());
			transform(leaves.begin(), leaves.end(), back_inserter(leaves_pred), [](const ptree_node& node) {
				return node->prediction;
				});
			leaves_ptr.reserve(leaves.size());
			for (const auto& lp : leaves) {
				leaves_ptr.emplace_back(lp);
			}
		}

		void Undo() override {
			auto tree = this->source_tree.lock();
			auto node = this->source.lock();

			tree->update_child_node_obs(node, rule);

			const auto nleaf = leaves_ptr.size();
			for (size_t li = 0; li < nleaf; li++) {
				if (leaves_ptr[li].expired())
					throw runtime_error("트리 구조가 바뀌었습니다");

				leaves_ptr[li].lock()->prediction = leaves_pred[li];
			}
		}

		void Undo_only_rule() const {
			auto node = this->source.lock();
			node->rule = rule;
		}

		void Undo_obs_rule() const {
			auto tree = this->source_tree.lock();
			auto node = this->source.lock();
			tree->update_child_node_obs(node, rule);
		}

		mcmc_patch patch() const override {
			auto tree = this->source_tree.lock();
			auto node = this->source.lock();
			auto leaves = tree->leaves_of_node(node);

			mcmc_patch patch;
			patch.reserve(leaves.size());
			for (size_t i = 0; i < leaves.size(); i++) {
				patch.push_back(pred_transaction{leaves_pred[i], leaves[i]->prediction, leaves[i]->obs_index});
			}
			return patch;
		}

		const split_rule& get_rule() const {
			return rule;
		}
	};

	MOVE select_mcmc_move() {
		return static_cast<MOVE>(random_utils::sample::index(MOVE_END_NUM));
	}

	void sample_node(const ptree_node& node) {
		node->rule.clear();
		node->prediction = sample_node_pred(node);
	}

	double proposal_tree_grow_lik_ratio(const ptree_node& node) const {
		double grow0 = prior_alpha * pow(node->depth, -prior_beta);
		return log(grow0) + 2 * log1p(-prior_alpha * pow(node->depth + 1, -prior_beta)) - log1p(-grow0);
	}

	double transition_rule_select_lik(const ptree& tree, const ptree_node& node) const {
		return transition_rule_select_lik(tree, node, node->rule);
	}

	double transition_rule_select_lik(const ptree& tree, const ptree_node& node, const split_rule& rule) const {
		double lik = log(static_cast<size_t>(mat_X.ncol()));
		if (rule.is_category(mat_X)) {
			lik += 0.69314718*tree->count_rule_selection(node, rule);
		} else {
			lik += log(tree->count_rule_selection(node, rule));
		}
		return lik;
	}

	move_result move_grow(const ptree& tree_proposed) {
		size_t count_growable;
		auto grow_cadidates = tree_proposed->select_growable_node_by_depth(prior_alpha, prior_beta, 3, &count_growable);

		ptree_node success_node;
		unique_ptr<MCMC_Grow> rollback_point;
		for (const auto& node : grow_cadidates) {
			rollback_point = make_unique<MCMC_Grow>(tree_proposed, node);
			try {
				auto split = tree_proposed->find_split_rule(node);
				tree_proposed->split_node(node, split);
				success_node = move(node);
				break;
			}
			catch (logic_error& e) {
				continue;
			}
		}

		if (!success_node)
			throw out_of_range("Can't find grow proposal");

		//Sampling node pred
		sample_node(success_node->left.lock());
		sample_node(success_node->right.lock());

		const double rjmcmc_grow_lr = log(count_growable)+transition_rule_select_lik(tree_proposed, success_node)-log(tree_proposed->count_prunable());
		const double data_lik_ratio = node_likelihood(success_node->left.lock()) + node_likelihood(success_node->right.lock()) - node_likelihood(success_node);
		rollback_point->transition_lik = rjmcmc_grow_lr + data_lik_ratio + proposal_tree_grow_lik_ratio(success_node);
		return rollback_point;
	}

	move_result move_prune(const ptree& tree_proposed) {
		size_t count_prunable;
		auto prune_cadidates = tree_proposed->select_prunable_node_by_depth(prior_alpha, prior_beta, 3, &count_prunable);
		
		ptree_node success_node;
		unique_ptr<MCMC_Prune> rollback_point;
		for (const auto& node : prune_cadidates) {
			rollback_point = make_unique<MCMC_Prune>(tree_proposed, node);
			rollback_point->transition_lik = -(node_likelihood(node->left.lock()) + node_likelihood(node->right.lock()));
			try {
				tree_proposed->prune_node(node);
				success_node = move(node);
				break;
			}
			catch (logic_error& e) {
				continue;
			}
		}

		if (!success_node)
			throw out_of_range("Can't find prune proposal");

		//Sampling node pred
		sample_node(success_node);

		const double rjmcmc_prune_lr = log(count_prunable)-transition_rule_select_lik(tree_proposed, success_node, rollback_point->get_rule())-log(tree_proposed->count_growable());
		const double data_lik_ratio_extra = node_likelihood(success_node);

		rollback_point->transition_lik += rjmcmc_prune_lr + data_lik_ratio_extra - proposal_tree_grow_lik_ratio(success_node);
		return rollback_point;
	}

	move_result move_change(const ptree& tree_proposed) {
		auto change_cadidates = tree_proposed->select_changeable_node(5);
		if (change_cadidates.size() == 0)
			throw out_of_range("Can't find change proposal");
		
		ptree_node success_node;
		unique_ptr<MCMC_Change> rollback_point;
		
		array<typename tree_structure_type::tree_node*, 3> arr_key;
		arr_key.fill(nullptr);
		array<pair<vector<ptree_node>, double>, 3> arr_leaf;
		arr_leaf.fill(make_pair(vector<ptree_node>(), 0.0));

		double current_node_lik = 0.0;
		vector<ptree_node> current_leaves;
		size_t last_insert = arr_key.size();

		for (const auto& node : change_cadidates) {
			if (last_insert < arr_key.size()) {
				bool is_independent = arr_key[last_insert]->depth <= node->depth; //깊이가 같거나 더 깊어지면 더이상 prev node를 소유하지 않음
				const auto& w_parent = arr_key[last_insert]->parent;
				if (!is_independent && !w_parent.expired()) //skip으로 더 위쪽 노드로 이동했다면 건너편 노드여도 깊이가 더 작을 수 있음
				{
					ptree_node cur = w_parent.lock();
					while (cur->depth > node->depth && !cur->parent.expired()) cur = cur->parent.lock();
					is_independent = cur != node; //현재 노드와 같은 높이까지 타고 올라왔을 때, prev node 조상이 현재 노드와 다르다면 독립이다
				}
				if (is_independent)
					rollback_point->Undo_obs_rule();
				else //현재 노드가 prev node의 삼위 노드인 경우
					rollback_point->Undo_only_rule(); //오류 발생. Undo_obs_rule();은 정상 작동
			}

			//현재 노드의 leaf를 탐색. 탐색 대상 노드는 항상 prev_node의 하위 노드가 아니다.
			{
				queue<ptree_node> search;
				search.push(node);
				while (!search.empty()) {
					const auto& front = search.front();
					if (front->isLeaf()) {
						current_node_lik += node_likelihood(front);
						current_leaves.push_back(move(front));
					}
					else {
						size_t i;
						for (i = 0; i < arr_key.size(); i++) {
							if (arr_key[i] == front.get()) {
								arr_key[i] = nullptr;
								
								current_leaves.insert(current_leaves.end(), arr_leaf[i].first.begin(), arr_leaf[i].first.end());
								arr_leaf[i].first.clear();

								current_node_lik += arr_leaf[i].second;
								arr_leaf[i].second = 0.0;
								break;
							}
						}
						if (i == arr_key.size()) {
							search.push(front->left.lock());
							search.push(front->right.lock());
						}
					}
					search.pop();
				}
			}
			rollback_point = make_unique<MCMC_Change>(tree_proposed, node, current_leaves);
			//find_split_rule은 해당 노드에서만 유효함을 보장한다. 따라서 하위 노드는 여러번 무작위 시도로 확인함.
			const auto max_try = min(static_cast<size_t>(50), node->obs_index.size());
			size_t i;
			for (i = 0; i < max_try; i++) {
				try {
					const auto split = tree_proposed->find_split_rule(node);
					if (tree_proposed->update_child_node_obs(node, split)) {
						break;
					}
				} catch (logic_error& e) {
					continue;
				}
			}

			if (i < max_try) {
				success_node = move(node);
				break;
			}

			last_insert = 0;
			while (arr_key[last_insert] == nullptr) last_insert++;
			arr_key[last_insert] = node.get();
			arr_leaf[last_insert].first = move(current_leaves); current_leaves.clear();
			arr_leaf[last_insert].second = current_node_lik;
		}

		if (!success_node) {
			//마지막 change_cadidates의 노드는 다음 for문에서 undo를 시켜주지 않는다. 따라서 여기서 처리.
			rollback_point->Undo_obs_rule();
			throw out_of_range("Can't find change proposal");
		}

		//Sampling node pred
		for (const auto& leaf : current_leaves) {
			sample_node(leaf);
		}
		
		double&& lik_ratio = accumulate(current_leaves.cbegin(), current_leaves.cend(), 0.0, [this](const double& sum, const ptree_node& node) {
			return sum+node_likelihood(node);
			});
		
		double&& rjmcmc_change_lik = transition_rule_select_lik(tree_proposed, success_node)-
										transition_rule_select_lik(tree_proposed, success_node, rollback_point->get_rule());
		
		rollback_point->transition_lik = rjmcmc_change_lik + lik_ratio - current_node_lik;
		return rollback_point;
	}

protected:
	const DataMatrix& get_predictor_mat() const {
		return mat_X;
	}

	array<size_t, MOVE_END_NUM> mcmc_accept;

	virtual typeTreeVal sample_node_pred(const ptree_node& node) = 0;
	virtual double node_likelihood(const ptree_node& node) const = 0;

	static bool is_accept(const double& log_lik_proposal) {
		const double accept_bound = log(random_utils::sample::uniform());
		return !std::isnan(log_lik_proposal) && (accept_bound <= log_lik_proposal);
	}

	pair<bool, mcmc_patch> MCMC_tree_move(const ptree& tree) {
		auto move = tree->is_stump() ? MOVE_GROW : select_mcmc_move();
		move_result mcmc_proposal;
		try {
			switch (move) {
			case MOVE_GROW: {
				mcmc_proposal = std::move(move_grow(tree));
				break;
			}
			case MOVE_PRUNE: {
				mcmc_proposal = std::move(move_prune(tree));
				break;
			}
			case MOVE_CHANGE: {
				mcmc_proposal = std::move(move_change(tree));
				break;
			}
			default:
			  throw out_of_range("정의되지 않은 Move");
			}
		}
		catch (const out_of_range& e) {
			return make_pair(false, mcmc_patch{});
		}

		if (is_accept(mcmc_proposal->transition_lik)) {
			//accept
			mcmc_accept[move]++;
			return make_pair(true, mcmc_proposal->patch());
		}
		//reject
		mcmc_proposal->Undo();
		return make_pair(false, mcmc_patch{});
	}

public:
	BART_base() = delete;
	BART_base(const BART_base&) = delete;
	virtual ~BART_base() = default;
	BART_base(const DataMatrix& X, size_t n_trees, double alpha=0.95, double beta=1) : mat_X(X) {
		mcmc_accept.fill(0);
		
		set_tree_structure_prior(alpha, beta);

		trees.reserve(n_trees);
		for (size_t i = 0; i < n_trees; i++) {
			auto tree = make_shared<tree_structure_type>(mat_X);
			trees.push_back(move(tree));
		}
	}

	void set_tree_structure_prior(double a, double b) {
		assert(0 < a && a <= 1);
		prior_alpha = a;
		assert(0 < b);
		prior_beta = b;
	}

	//Grow, Prune, Change 순
	auto accept_count() const {
		return mcmc_accept;
	}

	void clear_count() {
		mcmc_accept.fill(0);
	}
};

template<typename typeTreeVal>
class BART_serializable {
	using store_type = CTreeStore<typeTreeVal>;

	unique_ptr<store_type> store_instance;
	store_type* store;

public:
	BART_serializable() {
		store_instance = make_unique<store_type>();
		store = store_instance.get();
	}

	BART_serializable(size_t n_tree) {
		init_tree_store(n_tree);
	}

	void set_tree_store(store_type* new_store) {
		if (!new_store)
			return;
		if (store_instance)
			store_instance.reset();
		store = new_store;
	}

	void init_tree_store(size_t n_tree) {
		store_instance = make_unique<store_type>(n_tree);
		store = store_instance.get();
	}

	size_t n_bart() const {
		return store->internal_storage().size();
	}

	store_type* get_store() const {
		return store;
	}

	virtual ~BART_serializable() = default;
};

template<typename typeTreeVal, typename typePredVal = typeTreeVal>
class BART_predictable : protected BART_serializable<typeTreeVal> {
public:
	BART_predictable() = delete;
	BART_predictable(size_t n_tree) : BART_serializable<typeTreeVal>(n_tree) {}
	virtual ~BART_predictable() = default;

	struct BART_predict {
		vector<typePredVal> post_mean, ci_low, ci_up;
		BART_predict() = default;
		BART_predict(size_t nobs) {
			reserve(nobs);
		}

		void reserve(size_t nobs) {
			post_mean.reserve(nobs);
			ci_low.reserve(nobs);
			ci_up.reserve(nobs);
		}
	};

	virtual BART_predict predict(const DataMatrix& mat_X, double ci_interval=0.95) const = 0;
};

template<typename typeNumPredVal>
class BART_predictable_numeric : public BART_predictable<typeNumPredVal, typeNumPredVal> {
protected:
	vector<typeNumPredVal> serialized_predict(const DataMatrix& mat_X) const {
		// input: mat(n, p), output: vector(n*nbart), sutructure: [obs#1: len=nbart][obs#2: len=nbart]...[obs#n: len=nbart]
		const auto& store = this->get_store()->internal_storage();
		const size_t nobs = mat_X.nrow(), nbart = this->n_bart();
		if (nbart == 0)
			throw runtime_error("저장된 트리가 없습니다");
		vector<double> preds(nobs * nbart, 0.0);
		for (size_t group_index = 0; group_index < nbart; group_index++) {
			const auto& tree_group = store.at(group_index);
			for (size_t tree_idx = 0; tree_idx < tree_group.size(); tree_idx++) {
				auto&& tree_pred = tree_group.at(tree_idx)->predict(mat_X);
				for (size_t i = 0; i < nobs; i++)
					preds.at(i * nbart + group_index) += tree_pred.at(i);
			}
		}
		return preds;
	}
public:
	BART_predictable_numeric(size_t n_tree) : BART_predictable<typeNumPredVal, typeNumPredVal>(n_tree) {}
	using BART_predict = typename BART_predictable<typeNumPredVal, typeNumPredVal>::BART_predict;
	
	virtual BART_predict predict(const DataMatrix& mat_X, double ci_interval=0.95) const override {
		const size_t nobs = mat_X.nrow(), nbart = this->n_bart();
		auto preds = serialized_predict(mat_X);
		
		if (preds.size() != nbart*nobs)
			throw runtime_error("예측 크기 불일치");

		if (nbart == 1) {
			BART_predict pred(0);
			pred.ci_low = preds;
			pred.ci_up = preds;
			pred.post_mean = preds;
			return pred;
		}

		const double alpha = (1 - ci_interval) / 2, quantile_low = (nbart - 1) * alpha, quantile_up = (nbart - 1) * (1 - alpha);
		const size_t low_idx = static_cast<size_t>(floor(quantile_low)), up_idx = static_cast<size_t>(ceil(quantile_up));
		const double q_inter = quantile_low - low_idx;

		BART_predict pred(nobs);
		auto start = preds.begin(), end = start + nbart;
		for (size_t i = 0; i < nobs; i++) {
			sort(start, end);

			pred.ci_low.push_back(*(start + low_idx) * (1 - q_inter) + *(start + low_idx + 1) * q_inter);
			pred.ci_up.push_back(*(start+up_idx) * q_inter + *(start + up_idx - 1) * (1-q_inter));
			pred.post_mean.push_back(accumulate(start, end, 0.0)/nbart);

			start += nbart;
			end += nbart;
		}

		return pred;
	}
};