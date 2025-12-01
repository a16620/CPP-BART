#pragma once
#include <vector>
#include <memory>
#include <iostream>

class DataMatrix {
public:
    using numeric_type = double;
    using category_type = int;
private:
    std::vector<numeric_type> p_numeric_mat;
    std::vector<category_type> p_categorical_mat, p_max_category_index;
    size_t n_row, col_num, col_cat;
public:
    DataMatrix() : n_row(0), col_num(0), col_cat(0) {
    }

    DataMatrix(size_t row, size_t num_col, size_t cat_col) : n_row(row) {
        set_size_numeric(num_col);
        set_size_categorical(cat_col);
    }

    DataMatrix(const DataMatrix& om) : p_numeric_mat(om.p_numeric_mat), p_categorical_mat(om.p_categorical_mat), p_max_category_index(om.p_max_category_index),
                                        n_row(om.n_row), col_num(om.col_num), col_cat(om.col_cat) {}

    DataMatrix(DataMatrix&& om) : p_numeric_mat(std::move(om.p_numeric_mat)), p_categorical_mat(std::move(om.p_categorical_mat)), p_max_category_index(std::move(om.p_max_category_index)),
                                        n_row(om.n_row), col_num(om.col_num), col_cat(om.col_cat) {}

    DataMatrix(const std::vector<numeric_type>& col_vec) : p_numeric_mat(col_vec), n_row(col_vec.size()), col_num(1), col_cat(0) {
    }

    DataMatrix& operator=(const DataMatrix& o) {
        this->col_cat = o.col_cat;
        this->col_num = o.col_num;
        this->n_row = o.n_row;
        this->p_numeric_mat = o.p_numeric_mat;
        this->p_categorical_mat = o.p_categorical_mat;
        this->p_max_category_index = o.p_max_category_index;
        return *this;
    }

    DataMatrix& operator=(DataMatrix&& o) {
        this->col_cat = o.col_cat;
        this->col_num = o.col_num;
        this->n_row = o.n_row;
        this->p_numeric_mat = std::move(o.p_numeric_mat);
        this->p_categorical_mat = std::move(o.p_categorical_mat);
        this->p_max_category_index = std::move(o.p_max_category_index);
        return *this;
    }

    void set_size_numeric(size_t col) {
        col_num = col;

        const size_t total_len = n_row*col_num;
        if (total_len == 0) {
            clear_numeric();
            return;
        }
        p_numeric_mat.assign(total_len, 0.0);
    }

    void set_size_categorical(size_t col) {
        col_cat = col;

        const size_t total_len = n_row*col_cat;
        if (total_len == 0) {
            clear_categorical();
            return;
        }
        p_categorical_mat.assign(total_len, 0.0);
    }

    void set_size_row(size_t nrow) {
        this->n_row = nrow;
        set_size_numeric(col_num);
        set_size_categorical(col_cat);
    }

    auto raw_num() const {
        return p_numeric_mat.data();
    }

    auto raw_cat() const {
        return p_categorical_mat.data();
    }

    auto& raw_num_vec() {
        return p_numeric_mat;
    }

    auto& raw_cat_vec() {
        return p_categorical_mat;
    }

    auto& raw_cat_idx_vec() {
        return p_max_category_index;
    }

    void clear_numeric() {
        col_num = 0;
        p_numeric_mat.clear();
        if (col_cat == 0)
            n_row = 0;
    }

    void clear_categorical() {
        col_cat = 0;
        p_categorical_mat.clear();
        if (col_num == 0)
            n_row = 0;
    }

    void clear() {
        clear_numeric();
        clear_categorical();
    }

    void append_numeric(const std::vector<numeric_type>& ov) {
        if (ov.size() != n_row)
            throw std::invalid_argument("ov.size() != n_row");
        col_num++;
        p_numeric_mat.reserve(p_numeric_mat.size() + n_row);
        std::copy(ov.begin(), ov.end(), std::back_inserter(p_numeric_mat));
    }

    void append_numeric(numeric_type val) {
        col_num++;
        const size_t original_size = p_numeric_mat.size();
        p_numeric_mat.resize(original_size + n_row, val);
        //std::fill(p_numeric_mat.begin()+original_size, p_numeric_mat.end(), val);
    }

    void append_categorical(const std::vector<category_type>& ov, category_type end_index=0) {
        if (ov.size() != n_row)
            throw std::invalid_argument("ov.size() != n_row");
        col_num++;
        p_categorical_mat.reserve(p_categorical_mat.size() + n_row);
        std::copy(ov.cbegin(), ov.cend(), std::back_inserter(p_categorical_mat));
        if (end_index == 0)
            end_index = *std::max_element(ov.cbegin(), ov.cend());
        p_max_category_index.push_back(end_index);
    }

    bool is_col_vector() const {
        return col_num == 1 && col_cat == 0;
    }

    size_t nrow() const {
        return n_row;
    }

    size_t ncol_numeric() const {
        return col_num;
    }

    size_t ncol_categorical() const {
        return col_cat;
    }

    size_t ncol() const {
        return col_num+col_cat;
    }

    numeric_type& at_numeric(size_t i) {
        return p_numeric_mat.at(i);
    }

    category_type& at_categorical(size_t i) {
        return p_categorical_mat.at(i);
    }

    numeric_type& at_numeric(size_t col, size_t row) {
        return p_numeric_mat.at(n_row*col+row);
    }

    category_type& at_categorical(size_t col, size_t row) {
        return p_categorical_mat.at(n_row*col+row);
    }

    numeric_type at_numeric(size_t i) const {
        return p_numeric_mat.at(i);
    }

    category_type at_categorical(size_t i) const {
        return p_categorical_mat.at(i);
    }

    numeric_type at_numeric(size_t col, size_t row) const {
        return p_numeric_mat.at(n_row*col+row);
    }

    category_type at_categorical(size_t col, size_t row) const {
        return p_categorical_mat.at(n_row*col+row);
    }

    template<typename ret_as = numeric_type>
    ret_as at(size_t i) const {
        if (i >= p_numeric_mat.size()) {
            return p_categorical_mat.at(i - p_numeric_mat.size());
        } else {
            return p_numeric_mat.at(i);
        }
    }

    template<typename ret_as = numeric_type>
    ret_as at(size_t col, size_t row) const {
        if (col >= col_num) {
            return at_categorical(col - col_num, row);
        } else {
            return at_numeric(col, row);
        }
    }

    category_type max_index(size_t col) const {
        if (!is_category(col))
            throw std::invalid_argument("it is numerical column");
        return p_max_category_index.at(col-col_num);
    }

    bool is_category(size_t col) const {
        return col >= col_num;
    }

    DataMatrix as_full_numeric() const {
        const size_t total_dummy_count = std::accumulate(p_max_category_index.begin(), p_max_category_index.end(), static_cast<size_t>(0));
        DataMatrix dm(n_row, col_num + total_dummy_count, 0);
        std::copy(p_numeric_mat.cbegin(), p_numeric_mat.cend(), dm.raw_num_vec().begin());
        size_t dummy_offset = 0;
        for (size_t cat_col_idx = 0; cat_col_idx < col_cat; cat_col_idx++) {
            for (size_t row = 0; row < n_row; row++) {
                dm.at_numeric(dummy_offset+this->at_categorical(cat_col_idx, row)) = 1;
            }
            dummy_offset += p_max_category_index.at(cat_col_idx);
        }
        return dm;
    }

    DataMatrix operator*(const DataMatrix& o) const {
        if (col_num != o.n_row)
            throw std::invalid_argument("DataMatrix operator*: this->ncol() != o.nrow()");
        DataMatrix dm(n_row, o.col_num, 0);
        for (size_t r = 0; r < n_row; r++) {
            for (size_t c = 0; c < o.col_num; c++) {
                numeric_type sum = static_cast<numeric_type>(0.0);
                for (size_t i = 0; i < col_num; i++) {
                    sum += at_numeric(i, r)*o.at_numeric(c, i);
                }
                dm.at_numeric(c, r) = sum;
            }
        }
        return dm;
    }

    DataMatrix operator+(const DataMatrix& o) const {
        if (n_row != o.n_row)
            throw std::invalid_argument("DataMatrix operator-: n_row != o.n_row");
        
        const DataMatrix* shorter, *longer;
        size_t max_col, min_col;
        if (col_num < o.col_num) {
            shorter = this;
            longer = &o;
            max_col = o.col_num;
            min_col = col_num;
        } else {
            shorter = &o;
            longer = this;
            max_col = col_num;
            min_col = o.col_num;
        }

        DataMatrix dm(n_row, max_col, 0);
        for (size_t out_c = 0, inner_c = 0; out_c < max_col; out_c++, inner_c++) {
            if (inner_c == min_col) inner_c = 0;
            for (size_t i = 0; i < n_row; i++) {
                dm.at_numeric(out_c, i) = longer->at_numeric(out_c, i) + shorter->at_numeric(inner_c, i);
            }
        }
        return dm;
    }

    DataMatrix operator-() const {
        DataMatrix dm(*this);
        auto& raw_arr = dm.raw_num_vec();
        std::transform(raw_arr.begin(), raw_arr.end(), raw_arr.begin(), [](const numeric_type& v) {
            return -v;
        });
        return dm;
    }

    DataMatrix operator-(const DataMatrix& o) const {
        return (*this) + (-o);
    }

    DataMatrix transpose() const {
        DataMatrix dm(col_num, n_row, 0);
        for (size_t r = 0; r < col_num; r++) {
            for (size_t c = 0; c < n_row; c++) {
                dm.at_numeric(c, r) = this->at_numeric(r, c);
            }
        }
        return dm;
    }

    DataMatrix Hat_by_cholesky() const {
        using namespace std;
        DataMatrix xtx = this->transpose() * (*this);
        const auto N = xtx.n_row;
        {
            //Cholesky: L'L = x'x
            DataMatrix chol_L(N, N, 0);
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j <= i; j++) {
                    numeric_type sum = static_cast<numeric_type>(0.0);
                    for (size_t k = 0; k < j; k++)
                        sum += chol_L.at_numeric(k, i) * chol_L.at_numeric(k, j);

                    if (i == j)
                        chol_L.at_numeric(j, i) = sqrt(xtx.at_numeric(i ,i) - sum);
                    else
                        chol_L.at_numeric(j, i) = (xtx.at_numeric(j ,i) - sum) / chol_L.at_numeric(j, j);
                }
            }
            //inv
            xtx.set_size_numeric(N);
            for (size_t i = 0; i < N; i++) {
                xtx.at_numeric(i ,i) = 1.0 / chol_L.at_numeric(i, i);
                for (size_t j = 0; j < i; j++) {
                    numeric_type sum = static_cast<numeric_type>(0.0);
                    for (size_t k = j; k < i; k++) {
                        sum -= chol_L.at_numeric(k, i) * xtx.at_numeric(j, k);
                    }
                    xtx.at_numeric(j, i) = sum / chol_L.at_numeric(i, i);
                }
            }
            // = L_inv' * L_inv = inv(x'x)
            xtx = xtx.transpose() * xtx;
        }
        // H = x * inv(x'x) * x'
        return (*this) * xtx * this->transpose();
    }

    static DataMatrix identity(size_t n, numeric_type fill = 1.0) {
        DataMatrix dm(n, n, 0);
        for (size_t i = 0; i < n; i++) {
            dm.at_numeric(i, i) = fill;
        }
        return dm;
    }
};

