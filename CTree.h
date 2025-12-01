#pragma once
#include <numeric>
#include <memory>
#include <vector>
#include <queue>
#include <stack>
#include <set>
#include <unordered_set>
#include "random_utils.h"
#include "stat_utils.h"
#include "data_matrix.h"
using namespace std;

template<typename typeNodeVal>
class CTree
{
	static constexpr size_t minimum_obs_in_node = 3;
	static constexpr size_t minimum_obs_growable = 2*minimum_obs_in_node+1;
public:
	struct split_rule {
		size_t column;
		double split_point;
		std::set<DataMatrix::category_type> filter_class;

		split_rule() : column(0), split_point(0.0) {
		}

		split_rule(const split_rule& o) : column(o.column), split_point(o.split_point), filter_class(o.filter_class) {
		}

		split_rule(split_rule&& o) : column(o.column), split_point(o.split_point), filter_class(move(o.filter_class)) {
		}

		split_rule(size_t column, double split_point) : column(column), split_point(split_point) {
		}

		split_rule(size_t column, const set<DataMatrix::category_type>& filter_class) : column(column), filter_class(filter_class) {
		}

		split_rule(size_t column, set<DataMatrix::category_type>&& filter_class) : column(column), filter_class(move(filter_class)) {
		}

		split_rule& operator=(const split_rule& rule) {
			column = rule.column;
			filter_class = rule.filter_class;
			split_point = rule.split_point;
			return *this;
		}

		split_rule& operator=(const split_rule&& rule) {
			column = rule.column;
			filter_class = move(rule.filter_class);
			split_point = rule.split_point;
			return *this;
		}

		bool is_split_left(const DataMatrix& mat, size_t index) const {
			if (is_category(mat)) {
				return filter_class.count(mat.at<DataMatrix::category_type>(column, index)) > 0;
			}
			//first=left, second=right. X<=Rule.C ->left
			return mat.at_numeric(column, index) <= split_point;
		}

		bool is_category(const DataMatrix& mat) const {
			return mat.is_category(column);
		}

		void clear() {
			column = 0;
			split_point = 0.0;
			filter_class.clear();
		}
	};

	struct tree_node {
		vector<size_t> obs_index;
		weak_ptr<tree_node> left, right, parent;
		size_t depth = 0;

		split_rule rule;
		typeNodeVal prediction;

		bool isLeaf() const noexcept {
			return left.expired() && right.expired();
		}

		bool isGrowable() const {
			return isLeaf()  && obs_index.size() >= minimum_obs_growable;
		}

		bool isPrunable() const {
			return !isLeaf() && left.lock()->isLeaf() && right.lock()->isLeaf();
		}

		tree_node(const vector<size_t>& obs, weak_ptr<tree_node> parent) : obs_index(obs), parent(parent) {
		}

		tree_node(const vector<size_t>& obs) : obs_index(obs) {
		}

		tree_node(vector<size_t>&& obs, weak_ptr<tree_node> parent) : obs_index(obs), parent(parent) {
		}

		tree_node(vector<size_t>&& obs) : obs_index(obs) {
		}
	};


	bool update_child_node_obs(shared_ptr<tree_node> update_node, const split_rule& rule) {
		update_node->rule = rule;

		queue<shared_ptr<tree_node>> nodes;
		nodes.push(update_node);

		while (!nodes.empty()) {
			auto node = move(nodes.front());
			nodes.pop();

			if (!node->isLeaf()) {
				auto child_index = split_obs_index(node->obs_index, node->rule);
				if (child_index.first.size() == 0 || child_index.second.size() == 0)
					return false;

				auto left = node->left.lock(), right = node->right.lock();
				left->obs_index = move(child_index.first);
				right->obs_index = move(child_index.second);

				nodes.push(left);
				nodes.push(right);
			}
		}
		return true;
	}

	vector<shared_ptr<tree_node>> leaves_of_node(const shared_ptr<tree_node>& top_node) const {
		vector<shared_ptr<tree_node>> leaves;
		queue<shared_ptr<tree_node>> search;
		search.push(top_node);

		while (!search.empty()) {
			const auto& node = search.front();
			if (node->isLeaf()) {
				leaves.push_back(move(node));
			}
			else {
				search.push(node->left.lock());
				search.push(node->right.lock());
			}
			search.pop();
		}
		return leaves;
	}

	void split_node(const shared_ptr<tree_node>& leaf_node, const split_rule& rule) {
		auto child_index = split_obs_index(leaf_node->obs_index, rule);
		if (child_index.first.size() == 0 || child_index.second.size() == 0)
			throw logic_error("split rule is creating empty node");
		auto left = make_shared<tree_node>(move(child_index.first), leaf_node), right = make_shared<tree_node>(move(child_index.second), leaf_node);
		left->depth = leaf_node->depth + 1;
		right->depth = leaf_node->depth + 1;
		
		nodes.push_back(left);
		nodes.push_back(right);

		leaf_node->rule = rule;
		leaf_node->left = left;
		leaf_node->right = right;

		leaves.erase(remove_if(leaves.begin(), leaves.end(), [&leaf_node](const weak_ptr<tree_node>& comp_ptr) {
			return leaf_node == comp_ptr.lock();
			}), leaves.end());
		leaves.push_back(left);
		leaves.push_back(right);
	}

	void prune_node(const shared_ptr<tree_node>& leaf_node) {
		if (!leaf_node->isPrunable())
			throw logic_error("can't prune grandparent node");
			
		const auto left = leaf_node->left.lock(), right = leaf_node->right.lock();
		auto remove_pos_leaves = remove_if(leaves.begin(), leaves.end(), [&](const weak_ptr<tree_node>& wptr) {
			auto ptr = wptr.lock();
			return left == ptr || right == ptr;
			});
		leaves.erase(remove_pos_leaves, leaves.end());
		
		auto remove_pos_nodes = remove_if(nodes.begin(), nodes.end(), [&](const shared_ptr<tree_node>& ptr) {
			return left == ptr || right == ptr;
			});
		nodes.erase(remove_pos_nodes, nodes.end());

		leaf_node->left.reset();
		leaf_node->right.reset();
		leaves.push_back(leaf_node);
	}

	size_t count_growable() const {
		return count_if(leaves.begin(), leaves.end(), [](const weak_ptr<tree_node>& node) {
			return node.lock()->isGrowable();
			});
	}

	size_t count_prunable() const {
		return count_if(nodes.begin(), nodes.end(), [](const shared_ptr<tree_node>& node) {
			return node->isPrunable();
			});
	}

	size_t count_rule_selection(const shared_ptr<tree_node>& node) const {
		if (node->isLeaf())
			throw invalid_argument("node is leaf");
		const auto& rule = node->rule;
		if (rule.is_category(mat_obs)) {
			return extract_exist_category(node->obs_index, rule.column).size();
		}
		return node->obs_index.size();
	}

	size_t count_rule_selection(const shared_ptr<tree_node>& node, const split_rule& rule) const {
		if (rule.is_category(mat_obs)) {
			return extract_exist_category(node->obs_index, rule.column).size();
		}
		return node->obs_index.size();
	}

	vector<shared_ptr<tree_node>> select_growable_node_by_depth(double alpha, double beta, size_t n = 1, size_t *total_growable = nullptr) {
		vector<shared_ptr<tree_node>> growable;
		for (const auto& weak_node_ptr : leaves) {
			auto node = weak_node_ptr.lock();
			if (node->isGrowable())
				growable.push_back(move(node));
		}
		if (total_growable)
			*total_growable = growable.size();

		if (growable.size() <= 1)
			return growable;
		
		vector<double> prob;
		prob.reserve(growable.size());
		transform(growable.cbegin(), growable.cend(), back_inserter(prob), [=](const shared_ptr<tree_node>& node) {
			return pow(node->depth, -beta);
			});
		vector<shared_ptr<tree_node>> sampled;
		auto smp = random_utils::sample::weight_multiple_index(prob, min(n, growable.size()));
		for (const auto& idx : smp) {
			sampled.push_back(growable[idx]);
		}
		return sampled;
	}

	vector<shared_ptr<tree_node>> select_prunable_node_by_depth(double alpha, double beta, size_t n = 1, size_t *total_prunable=nullptr) {
		vector<shared_ptr<tree_node>> prunable;
		copy_if(nodes.cbegin(), nodes.cend(), back_inserter(prunable), [](const shared_ptr<tree_node>& node) {
			return node->isPrunable();
			});

		if (total_prunable)
			*total_prunable = prunable.size();

		if (prunable.size() <= 1)
			return prunable;

		vector<double> prob;
		prob.reserve(prunable.size());
		transform(prunable.begin(), prunable.end(), back_inserter(prob), [=](const shared_ptr<tree_node>& node) {
			return pow(node->depth, beta);
			});

		vector<shared_ptr<tree_node>> sampled;
		auto smp = random_utils::sample::weight_multiple_index(prob, min(n, prunable.size()));
		for (const auto& idx : smp) {
			sampled.push_back(prunable[idx]);
		}
		return sampled;
	}

	vector<shared_ptr<tree_node>> select_changeable_node(size_t n = 1, double p = 0.7) {
		if (p < 0 || p > 1)
			throw invalid_argument("제거 확률은 0과 1사이의 값이어야 합니다");

		map<size_t, size_t> depth_count;
		vector<shared_ptr<tree_node>> changeable;
		stack<shared_ptr<tree_node>> path;
		tree_node* last = nullptr;
		path.push(root.lock());

		while (!path.empty()) {
			const auto& top = path.top();
			if (top->isLeaf()) {
				last = top.get();
				//changeable.push_back(move(top));
				path.pop();
			}
			else {
				auto left = top->left.lock(), right = top->right.lock();
				if (last == left.get() || last == right.get()) { //만약 모든 하위 노드가 추가되었으면
					last = top.get();
					depth_count[last->depth]++;
					changeable.push_back(move(top));
					path.pop();
				}
				else { //랜덤으로 순회 방향을 선택
					if (random_utils::sample::bit()) {
						path.push(move(left));
						path.push(move(right));
					}
					else {
						path.push(move(right));
						path.push(move(left));
					}
				}
			}
		}
		
		if (changeable.size() <= 1)
			return changeable;

		size_t acc_dep = 0, dep_bound = depth_count.rbegin()->first;
		for (auto it = depth_count.begin(); it != depth_count.end(); it++) {
			acc_dep += random_utils::sample::binom(it->second, p);
			if (acc_dep >= n) {
				dep_bound = it->first;
				break;
			}
		}

		changeable.erase(remove_if(changeable.begin(), changeable.end(), [dep_bound](const shared_ptr<tree_node>& node) {
			return dep_bound < node->depth;
		}), changeable.end());

		vector<double> death_prob;
		transform(changeable.begin(), changeable.end(), back_inserter(death_prob), [](const shared_ptr<tree_node>& node) {
			return node->depth;
			});

		const size_t nvec = changeable.size();
		auto rem_idx_vec = random_utils::sample::weight_multiple_index(death_prob, min(nvec-n, nvec-1));
		unordered_set<size_t> remove_idx(rem_idx_vec.begin(), rem_idx_vec.end());
		size_t newIndex = 0;
		for (size_t i = 0; i < nvec; i++) {
			if (!remove_idx.count(i)) {
				changeable[newIndex++] = changeable[i];
			}
		}
		changeable.resize(newIndex);

		return changeable;
	}

	split_rule find_split_rule(const shared_ptr<tree_node>& leaf_node) {
		const auto random_col_order = random_utils::sample::multiple_index(mat_obs.ncol(), mat_obs.ncol());
		for (const size_t& col : random_col_order) {
			if (mat_obs.is_category(col)) {
				auto exist_class = extract_exist_category(leaf_node->obs_index, col);
				const auto n_class = exist_class.size();
				if (n_class <= 1) {
					continue;
				}

				return split_rule(col, random_utils::sample::random_subset(exist_class, random_utils::sample::weight_index(make_class_uniform_select_prob(n_class))+1));
			} else {
				auto split_values = extract_continuous_split_points(leaf_node->obs_index, col);
				const size_t count = split_values.size();
				if (count == 0)
					continue;
				else if (count == 1)
					return split_rule(col, split_values.front());
				
				//const auto sort_smp = minmax(split_values[random_utils::sample::index(count)], split_values[random_utils::sample::index(count)]);
				//return split_rule(col, random_utils::sample::uniform(sort_smp.first, sort_smp.second));
				return split_rule(col, stat_utils::quantile(split_values, random_utils::sample::uniform()));
			}
		}
		throw logic_error("Split을 찾을 수 없음(비었거나 같은 값만 있는 노드)");
	}

	shared_ptr<tree_node> get_root() const {
		return root.lock();
	}

	vector<shared_ptr<tree_node>> tree_leaves() const {
		vector<shared_ptr<tree_node>> leaves;
		leaves.reserve(this->leaves.size());
		for (const auto& wptr_leaf : this->leaves) {
			leaves.push_back(wptr_leaf.lock());
		}
		return leaves;
	}

	bool is_stump() const {
		return nodes.size() <= 1;
	}

	size_t max_depth() const {
		size_t max_val = 0;
		for (auto &wptr : leaves) {
			if (auto sp = wptr.lock()) { // 객체가 살아있을 때만
				max_val = max(max_val, sp->depth);
			}
		}
		return max_val;
	}

private:
	static vector<double> make_class_uniform_select_prob(size_t n) {
		vector<double> prob(n-1);
		for (size_t m = 1; m < n; m++) {
			prob[m] = 1 / std::tgamma(m+1) / std::tgamma(n-m+1);
		}
		return prob;
	}

	pair<vector<size_t>, vector<size_t>> split_obs_index(vector<size_t> index, const split_rule& rule) const {
		vector<size_t> left, right;
		for (const auto& idx : index) {
			if (rule.is_split_left(mat_obs, idx))
				left.push_back(idx);
			else
				right.push_back(idx);
		}
		return make_pair(left, right);
	}
	
	vector<double> extract_continuous_split_points(const vector<size_t> index, size_t col) const {
		vector<double> x_point(index.size());
		double max_value = mat_obs.at<double>(col, index.front());
		
		transform(index.begin(), index.end(), x_point.begin(), [this, col, &max_value](const size_t& idx) {
			return mat_obs.at<double>(col, idx);
			});
		sort(x_point.begin(), x_point.end());
		
		auto min_bound_it = x_point.begin(), max_bound_it = x_point.end()-1;
		double min_bound = *min_bound_it, max_bound = *max_bound_it;
		size_t min_n = 0, max_n = 0;
		while (min_n < minimum_obs_in_node && min_bound_it != max_bound_it) {
			if (*min_bound_it <= min_bound) {
				min_bound_it++;
				min_n++;
			} else {
				min_bound = *min_bound_it;
			}
		}
		
		while (max_n < minimum_obs_in_node && min_bound_it != max_bound_it) {
			if (*max_bound_it >= max_bound) {
				max_bound_it--;
				max_n++;
			} else {
				max_bound = *max_bound_it;
			}
		}

		if (max_n < minimum_obs_in_node || min_n < minimum_obs_in_node)
			return vector<double>();

		//x_point.erase(remove(x_point.begin(), x_point.end(), x_point.back()), x_point.end());
		auto ret = vector<double>(min_bound_it, max_bound_it+1);
		ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
		return ret;
	}

	set<DataMatrix::category_type> extract_exist_category(const vector<size_t> index, size_t col) const {
		auto max_index = mat_obs.max_index(col);
		set<DataMatrix::category_type> cls;
		for (const auto& idx : index) {
			cls.insert(mat_obs.at<DataMatrix::category_type>(col, idx));
			if (cls.size() == max_index)
				break;
		}
		return cls;
	}

	const DataMatrix& mat_obs;

	weak_ptr<tree_node> root;
	vector<shared_ptr<tree_node>> nodes;
	vector<weak_ptr<tree_node>> leaves;

public:
	CTree() = delete;
	CTree(const DataMatrix& obs) : mat_obs(obs)
	{
		vector<size_t> full_index(mat_obs.nrow());
		std::iota(full_index.begin(), full_index.end(), 0);
		auto root = make_shared<tree_node>(move(full_index));
		root->depth = 1;
		this->root = root;
		nodes.push_back(root);
		leaves.push_back(root);
	}

	CTree(const CTree& tree) : mat_obs(tree.mat_obs) {
		auto original_root = tree.root.lock();
		auto root = make_shared<tree_node>(original_root->obs_index);
		
		nodes.push_back(root);
		this->root = root;

		queue<pair<shared_ptr<tree_node>, shared_ptr<tree_node>>> nodes;
		nodes.push(make_pair(original_root, root));

		while (!nodes.empty()) {
			auto node = move(nodes.front());
			nodes.pop();

			if (node.first->isLeaf()) {
				node.second->prediction = node.first->prediction;
				leaves.push_back(node.second);
			}
			else {
				shared_ptr<tree_node> ori_left = node.first->left.lock(),
					ori_right = node.first->right.lock(),
					left = make_shared<tree_node>(ori_left->obs_index, node.second),
					right = make_shared<tree_node>(ori_right->obs_index, node.second);

				node.second->rule = node.first->rule;
				node.second->left = left;
				node.second->right = right;

				this->nodes.push_back(left);
				this->nodes.push_back(right);

				nodes.push(make_pair(ori_left, left));
				nodes.push(make_pair(ori_right, right));
			}
		}
	}
};

