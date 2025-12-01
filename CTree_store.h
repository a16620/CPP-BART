#pragma once
#include "CTree.h"
using namespace std;

template<typename typeNodeVal>
class CTreeCompressed {
	using rule_type = typename CTree<typeNodeVal>::split_rule;
	struct comp_tree_node {
		unique_ptr<comp_tree_node> left, right;
		rule_type rule;
		typeNodeVal prediction;

		bool isLeaf() const noexcept {
			return !left || !right;
		}
	};
	using ptree_node = unique_ptr<comp_tree_node>;
	ptree_node root;

	static pair<vector<size_t>, vector<size_t>> split_obs_index(const DataMatrix& mat_X, const vector<size_t> index, rule_type rule) {
		vector<size_t> left, right;
		for (const auto& idx : index) {
			if (rule.is_split_left(mat_X, idx))
				left.push_back(idx);
			else
				right.push_back(idx);
		}
		return make_pair(left, right);
	}
public:
	CTreeCompressed() = delete;
	CTreeCompressed(const CTreeCompressed& tree) {
		this->root = make_unique<comp_tree_node>();
		auto original_root = tree.root.get();

		queue<pair<comp_tree_node*, comp_tree_node*>> nodes;
		nodes.push(make_pair(original_root, root.get()));

		while (!nodes.empty()) {
			auto node = nodes.front();
			nodes.pop();

			if (node.first->isLeaf()) {
				node.second->prediction = node.first->prediction;
			}
			else {
				auto ori_left = node.first->left.get(),
					ori_right = node.first->right.get();

				node.second->left = make_unique<comp_tree_node>();
				node.second->right = make_unique<comp_tree_node>();
				node.second->rule = node.first->rule;

				nodes.push(make_pair(ori_left, node.second->left.get()));
				nodes.push(make_pair(ori_right, node.second->right.get()));
			}
		}
	}

	CTreeCompressed(const shared_ptr<CTree<typeNodeVal>>& tree) {
		this->root = make_unique<comp_tree_node>();
		auto original_root = tree->get_root().get();

		queue<pair<typename CTree<typeNodeVal>::tree_node*, comp_tree_node*>> nodes;
		nodes.push(make_pair(original_root, root.get()));

		while (!nodes.empty()) {
			auto node = nodes.front();
			nodes.pop();

			if (node.first->isLeaf()) {
				node.second->prediction = node.first->prediction;
			}
			else {
				auto ori_left = node.first->left.lock().get(),
					ori_right = node.first->right.lock().get();

				node.second->left = make_unique<comp_tree_node>();
				node.second->right = make_unique<comp_tree_node>();
				node.second->rule = node.first->rule;

				nodes.push(make_pair(ori_left, node.second->left.get()));
				nodes.push(make_pair(ori_right, node.second->right.get()));
			}
		}
	}

	vector<double> predict(const DataMatrix& X) const {
		vector<double> pred(X.nrow());
		queue <pair<comp_tree_node*, vector<size_t>>> split_queue;
		{
			vector<size_t> index(X.nrow());
			iota(index.begin(), index.end(), 0);
			split_queue.push(make_pair(root.get(), move(index)));
		}

		while (!split_queue.empty()) {
			auto node_obs = move(split_queue.front());
			split_queue.pop();

			if (node_obs.first->isLeaf()) {
				for (const auto& idx : node_obs.second) {
					pred[idx] = node_obs.first->prediction;
				}
			}
			else {
				auto split_obs = split_obs_index(X, node_obs.second, node_obs.first->rule);
				split_queue.push(make_pair(node_obs.first->left.get(), move(split_obs.first)));
				split_queue.push(make_pair(node_obs.first->right.get(), move(split_obs.second)));
			}
		}
		return pred;
	}
};

template<typename typeNodeVal>
class CTreeStore {	
public:
	using tree_type = CTree<typeNodeVal>;
	using tree_compressed_type = CTreeCompressed<typeNodeVal>;
	using tree_group = vector<shared_ptr<tree_compressed_type>>;

	CTreeStore() {
		capacity = 0;
	}
	CTreeStore(size_t n_tree) {
		set_capacity(n_tree);
	}

	void set_capacity(size_t n_tree) {
		capacity = n_tree;
		temp_storage.reserve(n_tree);
		temp_storage.clear();
	}

	const vector<tree_group>& internal_storage() const {
		return store;
	}

	void load_temporal(const vector<shared_ptr<tree_type>>& trees) {
		if (trees.size() != capacity) {
			if (capacity == 0)
				set_capacity(trees.size());
			else
				throw logic_error("트리 저장소 크기와 입력 크기가 맞아야 합니다");
		}

		if (temp_storage.size() > 0)
			temp_storage.clear();

		for (const auto& tree : trees) {
			temp_storage.push_back(make_unique<tree_compressed_type>(tree));
		}
	}

	void load_temporal(const shared_ptr<tree_type>& tree) {
		if (temp_storage.size() >= capacity)
			throw logic_error("가득찬 임시 저장소는 flush전에 이용할 수 없습니다");
		temp_storage.push_back(make_shared<tree_compressed_type>(tree));
	}

	void repeat_temporal(size_t original_index) {
		temp_storage.push_back(store.back().at(original_index));
	}

	void flush() {
		store.push_back(move(temp_storage));
		temp_storage.reserve(capacity);
		temp_storage.clear();
	}

private:
	vector<tree_group> store;
	tree_group temp_storage;
	size_t capacity;
};