#pragma once
#include "bart.h"
#include <Rcpp.h>

using namespace Rcpp;
using namespace std;

class DataFrameTransformer {
    map<string, size_t> attribute_map_numeric, attribute_map_categorical;    
    size_t n_predictor_numeric, n_predictor_categorical;

    vector<DataMatrix::category_type> category_max_index;
public:
    void fit(const Rcpp::DataFrame& r_df) {
        attribute_map_numeric.clear();
        attribute_map_categorical.clear();
        category_max_index.clear();
        n_predictor_numeric = n_predictor_categorical = 0;

        const auto col_names = as<CharacterVector>(r_df.names());
        assert(r_df.ncol() >= 0);
        const size_t col_len = static_cast<size_t>(r_df.ncol());
        for (size_t col = 0; col < col_len; col++) {
            string p_name = as<std::string>(col_names.at(col));
            RObject robj = r_df[col];

            if (Rf_isFactor(robj)) {
                IntegerVector iv = as<IntegerVector>(robj);
                auto max_index = *max_element(iv.begin(), iv.end())-1;
                category_max_index.push_back(max_index);
                attribute_map_categorical.insert(make_pair(p_name, n_predictor_categorical));
                n_predictor_categorical++;
            } else if (Rf_isNumeric(robj)) {
                attribute_map_numeric.insert(make_pair(p_name, n_predictor_numeric++));
            }
        }
    }

    DataMatrix transform(const Rcpp::DataFrame& r_df) const {
        DataMatrix dm(r_df.nrow(), n_predictor_numeric, n_predictor_categorical);
        
        auto* const ptr_num_arr = dm.raw_num_vec().data();
        auto* const ptr_cat_arr = dm.raw_cat_vec().data();
        auto& factor_vec = dm.raw_cat_idx_vec();
        factor_vec.assign(n_predictor_categorical, 0);
        
        const auto col_names = as<CharacterVector>(r_df.names());
        assert(r_df.ncol() >= 0);
        const size_t col_len = static_cast<size_t>(r_df.ncol());
        for (size_t col = 0; col < col_len; col++) {
            string p_name = as<std::string>(col_names.at(col));
            if (auto f = attribute_map_numeric.find(p_name); f != attribute_map_numeric.end()) {
                NumericVector nv = r_df[col];
                static_assert(is_same<DataMatrix::numeric_type, double>::value, "DataMatrix::numeric_type must be equal to double, or use std::copy instead memcpy");
                std::memcpy(ptr_num_arr+dm.nrow()*f->second, nv.begin(), sizeof(DataMatrix::numeric_type)*dm.nrow());
            } else if (auto f = attribute_map_categorical.find(p_name); f != attribute_map_categorical.end()) {
                IntegerVector iv = r_df[col];
                static_assert(is_same<DataMatrix::category_type, int>::value, "DataMatrix::category_type must be equal to int, or use std::copy instead memcpy");
                std::memcpy(ptr_cat_arr+dm.nrow()*f->second, iv.begin(), sizeof(DataMatrix::category_type)*dm.nrow());
                factor_vec.at(f->second) = category_max_index.at(f->second);
            }
        }

        return dm;
    }
};

class RcppBART {
    DataFrameTransformer transformer;
    unique_ptr<BART> p_bart;
public:
    RcppBART() = delete;
    RcppBART(const DataFrame& X, const NumericVector& Y, size_t m) {
        transformer.fit(X);
        p_bart = make_unique<BART>(transformer.transform(X), std::vector<double>(Y.begin(), Y.end()), m);
    }

    void mcmc(size_t bi, size_t re) {
        p_bart->mcmc(bi, re);
    }

    NumericVector predict(const DataFrame& X) const {
        const auto dm_X = transformer.transform(X);
        auto pred = p_bart->predict(dm_X);
        return NumericVector(pred.post_mean.begin(), pred.post_mean.end());
    }
};