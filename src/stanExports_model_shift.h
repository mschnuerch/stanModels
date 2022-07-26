// Generated by rstantools.  Do not edit by hand.

/*
    stanModels is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    stanModels is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with stanModels.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef MODELS_HPP
#define MODELS_HPP
#define STAN__SERVICES__COMMAND_HPP
#include <rstan/rstaninc.hpp>
// Code generated by Stan version 2.21.0
#include <stan/model/model_header.hpp>
namespace model_model_shift_namespace {
using std::istream;
using std::string;
using std::stringstream;
using std::vector;
using stan::io::dump;
using stan::math::lgamma;
using stan::model::prob_grad;
using namespace stan::math;
static int current_statement_begin__;
stan::io::program_reader prog_reader__() {
    stan::io::program_reader reader;
    reader.add_event(0, 0, "start", "model_model_shift");
    reader.add_event(42, 40, "end", "model_model_shift");
    return reader;
}
#include <stan_meta_header.hpp>
class model_model_shift
  : public stan::model::model_base_crtp<model_model_shift> {
private:
        int K;
        std::vector<int> y1;
        std::vector<int> y2;
        double b1;
        double b2;
public:
    model_model_shift(stan::io::var_context& context__,
        std::ostream* pstream__ = 0)
        : model_base_crtp(0) {
        ctor_body(context__, 0, pstream__);
    }
    model_model_shift(stan::io::var_context& context__,
        unsigned int random_seed__,
        std::ostream* pstream__ = 0)
        : model_base_crtp(0) {
        ctor_body(context__, random_seed__, pstream__);
    }
    void ctor_body(stan::io::var_context& context__,
                   unsigned int random_seed__,
                   std::ostream* pstream__) {
        typedef double local_scalar_t__;
        boost::ecuyer1988 base_rng__ =
          stan::services::util::create_rng(random_seed__, 0);
        (void) base_rng__;  // suppress unused var warning
        current_statement_begin__ = -1;
        static const char* function__ = "model_model_shift_namespace::model_model_shift";
        (void) function__;  // dummy to suppress unused var warning
        size_t pos__;
        (void) pos__;  // dummy to suppress unused var warning
        std::vector<int> vals_i__;
        std::vector<double> vals_r__;
        local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // suppress unused var warning
        try {
            // initialize data block variables from context__
            current_statement_begin__ = 2;
            context__.validate_dims("data initialization", "K", "int", context__.to_vec());
            K = int(0);
            vals_i__ = context__.vals_i("K");
            pos__ = 0;
            K = vals_i__[pos__++];
            check_greater_or_equal(function__, "K", K, 2);
            current_statement_begin__ = 3;
            validate_non_negative_index("y1", "K", K);
            context__.validate_dims("data initialization", "y1", "int", context__.to_vec(K));
            y1 = std::vector<int>(K, int(0));
            vals_i__ = context__.vals_i("y1");
            pos__ = 0;
            size_t y1_k_0_max__ = K;
            for (size_t k_0__ = 0; k_0__ < y1_k_0_max__; ++k_0__) {
                y1[k_0__] = vals_i__[pos__++];
            }
            size_t y1_i_0_max__ = K;
            for (size_t i_0__ = 0; i_0__ < y1_i_0_max__; ++i_0__) {
                check_greater_or_equal(function__, "y1[i_0__]", y1[i_0__], 0);
            }
            current_statement_begin__ = 4;
            validate_non_negative_index("y2", "K", K);
            context__.validate_dims("data initialization", "y2", "int", context__.to_vec(K));
            y2 = std::vector<int>(K, int(0));
            vals_i__ = context__.vals_i("y2");
            pos__ = 0;
            size_t y2_k_0_max__ = K;
            for (size_t k_0__ = 0; k_0__ < y2_k_0_max__; ++k_0__) {
                y2[k_0__] = vals_i__[pos__++];
            }
            size_t y2_i_0_max__ = K;
            for (size_t i_0__ = 0; i_0__ < y2_i_0_max__; ++i_0__) {
                check_greater_or_equal(function__, "y2[i_0__]", y2[i_0__], 0);
            }
            current_statement_begin__ = 5;
            context__.validate_dims("data initialization", "b1", "double", context__.to_vec());
            b1 = double(0);
            vals_r__ = context__.vals_r("b1");
            pos__ = 0;
            b1 = vals_r__[pos__++];
            check_greater_or_equal(function__, "b1", b1, 0);
            current_statement_begin__ = 6;
            context__.validate_dims("data initialization", "b2", "double", context__.to_vec());
            b2 = double(0);
            vals_r__ = context__.vals_r("b2");
            pos__ = 0;
            b2 = vals_r__[pos__++];
            check_greater_or_equal(function__, "b2", b2, 0);
            // initialize transformed data variables
            // execute transformed data statements
            // validate transformed data
            // validate, set parameter ranges
            num_params_r__ = 0U;
            param_ranges_i__.clear();
            current_statement_begin__ = 10;
            validate_non_negative_index("alpha", "(K - 1)", (K - 1));
            num_params_r__ += (K - 1);
            current_statement_begin__ = 11;
            num_params_r__ += 1;
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e, current_statement_begin__, prog_reader__());
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }
    }
    ~model_model_shift() { }
    void transform_inits(const stan::io::var_context& context__,
                         std::vector<int>& params_i__,
                         std::vector<double>& params_r__,
                         std::ostream* pstream__) const {
        typedef double local_scalar_t__;
        stan::io::writer<double> writer__(params_r__, params_i__);
        size_t pos__;
        (void) pos__; // dummy call to supress warning
        std::vector<double> vals_r__;
        std::vector<int> vals_i__;
        current_statement_begin__ = 10;
        if (!(context__.contains_r("alpha")))
            stan::lang::rethrow_located(std::runtime_error(std::string("Variable alpha missing")), current_statement_begin__, prog_reader__());
        vals_r__ = context__.vals_r("alpha");
        pos__ = 0U;
        validate_non_negative_index("alpha", "(K - 1)", (K - 1));
        context__.validate_dims("parameter initialization", "alpha", "vector_d", context__.to_vec((K - 1)));
        Eigen::Matrix<double, Eigen::Dynamic, 1> alpha((K - 1));
        size_t alpha_j_1_max__ = (K - 1);
        for (size_t j_1__ = 0; j_1__ < alpha_j_1_max__; ++j_1__) {
            alpha(j_1__) = vals_r__[pos__++];
        }
        try {
            writer__.ordered_unconstrain(alpha);
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(std::runtime_error(std::string("Error transforming variable alpha: ") + e.what()), current_statement_begin__, prog_reader__());
        }
        current_statement_begin__ = 11;
        if (!(context__.contains_r("theta")))
            stan::lang::rethrow_located(std::runtime_error(std::string("Variable theta missing")), current_statement_begin__, prog_reader__());
        vals_r__ = context__.vals_r("theta");
        pos__ = 0U;
        context__.validate_dims("parameter initialization", "theta", "double", context__.to_vec());
        double theta(0);
        theta = vals_r__[pos__++];
        try {
            writer__.scalar_unconstrain(theta);
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(std::runtime_error(std::string("Error transforming variable theta: ") + e.what()), current_statement_begin__, prog_reader__());
        }
        params_r__ = writer__.data_r();
        params_i__ = writer__.data_i();
    }
    void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream__) const {
      std::vector<double> params_r_vec;
      std::vector<int> params_i_vec;
      transform_inits(context, params_i_vec, params_r_vec, pstream__);
      params_r.resize(params_r_vec.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r(i) = params_r_vec[i];
    }
    template <bool propto__, bool jacobian__, typename T__>
    T__ log_prob(std::vector<T__>& params_r__,
                 std::vector<int>& params_i__,
                 std::ostream* pstream__ = 0) const {
        typedef T__ local_scalar_t__;
        local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // dummy to suppress unused var warning
        T__ lp__(0.0);
        stan::math::accumulator<T__> lp_accum__;
        try {
            stan::io::reader<local_scalar_t__> in__(params_r__, params_i__);
            // model parameters
            current_statement_begin__ = 10;
            Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, 1> alpha;
            (void) alpha;  // dummy to suppress unused var warning
            if (jacobian__)
                alpha = in__.ordered_constrain((K - 1), lp__);
            else
                alpha = in__.ordered_constrain((K - 1));
            current_statement_begin__ = 11;
            local_scalar_t__ theta;
            (void) theta;  // dummy to suppress unused var warning
            if (jacobian__)
                theta = in__.scalar_constrain(lp__);
            else
                theta = in__.scalar_constrain();
            // transformed parameters
            current_statement_begin__ = 15;
            validate_non_negative_index("gamma1", "(K - 1)", (K - 1));
            Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, 1> gamma1((K - 1));
            stan::math::initialize(gamma1, DUMMY_VAR__);
            stan::math::fill(gamma1, DUMMY_VAR__);
            current_statement_begin__ = 16;
            validate_non_negative_index("gamma2", "(K - 1)", (K - 1));
            Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, 1> gamma2((K - 1));
            stan::math::initialize(gamma2, DUMMY_VAR__);
            stan::math::fill(gamma2, DUMMY_VAR__);
            current_statement_begin__ = 17;
            validate_non_negative_index("lambda1", "K", K);
            Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, 1> lambda1(K);
            stan::math::initialize(lambda1, DUMMY_VAR__);
            stan::math::fill(lambda1, DUMMY_VAR__);
            current_statement_begin__ = 18;
            validate_non_negative_index("lambda2", "K", K);
            Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, 1> lambda2(K);
            stan::math::initialize(lambda2, DUMMY_VAR__);
            stan::math::fill(lambda2, DUMMY_VAR__);
            // transformed parameters block statements
            current_statement_begin__ = 20;
            for (int k = 1; k <= (K - 1); ++k) {
                current_statement_begin__ = 21;
                stan::model::assign(gamma2, 
                            stan::model::cons_list(stan::model::index_uni(k), stan::model::nil_index_list()), 
                            (get_base1(alpha, k, "alpha", 1) - (.5 * theta)), 
                            "assigning variable gamma2");
                current_statement_begin__ = 22;
                stan::model::assign(gamma1, 
                            stan::model::cons_list(stan::model::index_uni(k), stan::model::nil_index_list()), 
                            (get_base1(alpha, k, "alpha", 1) + (.5 * theta)), 
                            "assigning variable gamma1");
            }
            current_statement_begin__ = 25;
            stan::model::assign(lambda1, 
                        stan::model::cons_list(stan::model::index_uni(1), stan::model::nil_index_list()), 
                        Phi_approx(get_base1(gamma1, 1, "gamma1", 1)), 
                        "assigning variable lambda1");
            current_statement_begin__ = 26;
            stan::model::assign(lambda2, 
                        stan::model::cons_list(stan::model::index_uni(1), stan::model::nil_index_list()), 
                        Phi_approx(get_base1(gamma2, 1, "gamma2", 1)), 
                        "assigning variable lambda2");
            current_statement_begin__ = 27;
            for (int k = 2; k <= (K - 1); ++k) {
                current_statement_begin__ = 28;
                stan::model::assign(lambda1, 
                            stan::model::cons_list(stan::model::index_uni(k), stan::model::nil_index_list()), 
                            (Phi_approx(get_base1(gamma1, k, "gamma1", 1)) - Phi_approx(get_base1(gamma1, (k - 1), "gamma1", 1))), 
                            "assigning variable lambda1");
                current_statement_begin__ = 29;
                stan::model::assign(lambda2, 
                            stan::model::cons_list(stan::model::index_uni(k), stan::model::nil_index_list()), 
                            (Phi_approx(get_base1(gamma2, k, "gamma2", 1)) - Phi_approx(get_base1(gamma2, (k - 1), "gamma2", 1))), 
                            "assigning variable lambda2");
            }
            current_statement_begin__ = 31;
            stan::model::assign(lambda1, 
                        stan::model::cons_list(stan::model::index_uni(K), stan::model::nil_index_list()), 
                        (1 - Phi_approx(get_base1(gamma1, (K - 1), "gamma1", 1))), 
                        "assigning variable lambda1");
            current_statement_begin__ = 32;
            stan::model::assign(lambda2, 
                        stan::model::cons_list(stan::model::index_uni(K), stan::model::nil_index_list()), 
                        (1 - Phi_approx(get_base1(gamma2, (K - 1), "gamma2", 1))), 
                        "assigning variable lambda2");
            // validate transformed parameters
            const char* function__ = "validate transformed params";
            (void) function__;  // dummy to suppress unused var warning
            current_statement_begin__ = 15;
            size_t gamma1_j_1_max__ = (K - 1);
            for (size_t j_1__ = 0; j_1__ < gamma1_j_1_max__; ++j_1__) {
                if (stan::math::is_uninitialized(gamma1(j_1__))) {
                    std::stringstream msg__;
                    msg__ << "Undefined transformed parameter: gamma1" << "(" << j_1__ << ")";
                    stan::lang::rethrow_located(std::runtime_error(std::string("Error initializing variable gamma1: ") + msg__.str()), current_statement_begin__, prog_reader__());
                }
            }
            stan::math::check_ordered(function__, "gamma1", gamma1);
            current_statement_begin__ = 16;
            size_t gamma2_j_1_max__ = (K - 1);
            for (size_t j_1__ = 0; j_1__ < gamma2_j_1_max__; ++j_1__) {
                if (stan::math::is_uninitialized(gamma2(j_1__))) {
                    std::stringstream msg__;
                    msg__ << "Undefined transformed parameter: gamma2" << "(" << j_1__ << ")";
                    stan::lang::rethrow_located(std::runtime_error(std::string("Error initializing variable gamma2: ") + msg__.str()), current_statement_begin__, prog_reader__());
                }
            }
            stan::math::check_ordered(function__, "gamma2", gamma2);
            current_statement_begin__ = 17;
            size_t lambda1_j_1_max__ = K;
            for (size_t j_1__ = 0; j_1__ < lambda1_j_1_max__; ++j_1__) {
                if (stan::math::is_uninitialized(lambda1(j_1__))) {
                    std::stringstream msg__;
                    msg__ << "Undefined transformed parameter: lambda1" << "(" << j_1__ << ")";
                    stan::lang::rethrow_located(std::runtime_error(std::string("Error initializing variable lambda1: ") + msg__.str()), current_statement_begin__, prog_reader__());
                }
            }
            current_statement_begin__ = 18;
            size_t lambda2_j_1_max__ = K;
            for (size_t j_1__ = 0; j_1__ < lambda2_j_1_max__; ++j_1__) {
                if (stan::math::is_uninitialized(lambda2(j_1__))) {
                    std::stringstream msg__;
                    msg__ << "Undefined transformed parameter: lambda2" << "(" << j_1__ << ")";
                    stan::lang::rethrow_located(std::runtime_error(std::string("Error initializing variable lambda2: ") + msg__.str()), current_statement_begin__, prog_reader__());
                }
            }
            // model body
            current_statement_begin__ = 36;
            lp_accum__.add(normal_log(alpha, 0, b1));
            current_statement_begin__ = 37;
            lp_accum__.add(normal_log(theta, 0, b2));
            current_statement_begin__ = 38;
            lp_accum__.add(multinomial_log(y1, lambda1));
            current_statement_begin__ = 39;
            lp_accum__.add(multinomial_log(y2, lambda2));
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e, current_statement_begin__, prog_reader__());
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }
        lp_accum__.add(lp__);
        return lp_accum__.sum();
    } // log_prob()
    template <bool propto, bool jacobian, typename T_>
    T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
               std::ostream* pstream = 0) const {
      std::vector<T_> vec_params_r;
      vec_params_r.reserve(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        vec_params_r.push_back(params_r(i));
      std::vector<int> vec_params_i;
      return log_prob<propto,jacobian,T_>(vec_params_r, vec_params_i, pstream);
    }
    void get_param_names(std::vector<std::string>& names__) const {
        names__.resize(0);
        names__.push_back("alpha");
        names__.push_back("theta");
        names__.push_back("gamma1");
        names__.push_back("gamma2");
        names__.push_back("lambda1");
        names__.push_back("lambda2");
    }
    void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
        dimss__.resize(0);
        std::vector<size_t> dims__;
        dims__.resize(0);
        dims__.push_back((K - 1));
        dimss__.push_back(dims__);
        dims__.resize(0);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dims__.push_back((K - 1));
        dimss__.push_back(dims__);
        dims__.resize(0);
        dims__.push_back((K - 1));
        dimss__.push_back(dims__);
        dims__.resize(0);
        dims__.push_back(K);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dims__.push_back(K);
        dimss__.push_back(dims__);
    }
    template <typename RNG>
    void write_array(RNG& base_rng__,
                     std::vector<double>& params_r__,
                     std::vector<int>& params_i__,
                     std::vector<double>& vars__,
                     bool include_tparams__ = true,
                     bool include_gqs__ = true,
                     std::ostream* pstream__ = 0) const {
        typedef double local_scalar_t__;
        vars__.resize(0);
        stan::io::reader<local_scalar_t__> in__(params_r__, params_i__);
        static const char* function__ = "model_model_shift_namespace::write_array";
        (void) function__;  // dummy to suppress unused var warning
        // read-transform, write parameters
        Eigen::Matrix<double, Eigen::Dynamic, 1> alpha = in__.ordered_constrain((K - 1));
        size_t alpha_j_1_max__ = (K - 1);
        for (size_t j_1__ = 0; j_1__ < alpha_j_1_max__; ++j_1__) {
            vars__.push_back(alpha(j_1__));
        }
        double theta = in__.scalar_constrain();
        vars__.push_back(theta);
        double lp__ = 0.0;
        (void) lp__;  // dummy to suppress unused var warning
        stan::math::accumulator<double> lp_accum__;
        local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // suppress unused var warning
        if (!include_tparams__ && !include_gqs__) return;
        try {
            // declare and define transformed parameters
            current_statement_begin__ = 15;
            validate_non_negative_index("gamma1", "(K - 1)", (K - 1));
            Eigen::Matrix<double, Eigen::Dynamic, 1> gamma1((K - 1));
            stan::math::initialize(gamma1, DUMMY_VAR__);
            stan::math::fill(gamma1, DUMMY_VAR__);
            current_statement_begin__ = 16;
            validate_non_negative_index("gamma2", "(K - 1)", (K - 1));
            Eigen::Matrix<double, Eigen::Dynamic, 1> gamma2((K - 1));
            stan::math::initialize(gamma2, DUMMY_VAR__);
            stan::math::fill(gamma2, DUMMY_VAR__);
            current_statement_begin__ = 17;
            validate_non_negative_index("lambda1", "K", K);
            Eigen::Matrix<double, Eigen::Dynamic, 1> lambda1(K);
            stan::math::initialize(lambda1, DUMMY_VAR__);
            stan::math::fill(lambda1, DUMMY_VAR__);
            current_statement_begin__ = 18;
            validate_non_negative_index("lambda2", "K", K);
            Eigen::Matrix<double, Eigen::Dynamic, 1> lambda2(K);
            stan::math::initialize(lambda2, DUMMY_VAR__);
            stan::math::fill(lambda2, DUMMY_VAR__);
            // do transformed parameters statements
            current_statement_begin__ = 20;
            for (int k = 1; k <= (K - 1); ++k) {
                current_statement_begin__ = 21;
                stan::model::assign(gamma2, 
                            stan::model::cons_list(stan::model::index_uni(k), stan::model::nil_index_list()), 
                            (get_base1(alpha, k, "alpha", 1) - (.5 * theta)), 
                            "assigning variable gamma2");
                current_statement_begin__ = 22;
                stan::model::assign(gamma1, 
                            stan::model::cons_list(stan::model::index_uni(k), stan::model::nil_index_list()), 
                            (get_base1(alpha, k, "alpha", 1) + (.5 * theta)), 
                            "assigning variable gamma1");
            }
            current_statement_begin__ = 25;
            stan::model::assign(lambda1, 
                        stan::model::cons_list(stan::model::index_uni(1), stan::model::nil_index_list()), 
                        Phi_approx(get_base1(gamma1, 1, "gamma1", 1)), 
                        "assigning variable lambda1");
            current_statement_begin__ = 26;
            stan::model::assign(lambda2, 
                        stan::model::cons_list(stan::model::index_uni(1), stan::model::nil_index_list()), 
                        Phi_approx(get_base1(gamma2, 1, "gamma2", 1)), 
                        "assigning variable lambda2");
            current_statement_begin__ = 27;
            for (int k = 2; k <= (K - 1); ++k) {
                current_statement_begin__ = 28;
                stan::model::assign(lambda1, 
                            stan::model::cons_list(stan::model::index_uni(k), stan::model::nil_index_list()), 
                            (Phi_approx(get_base1(gamma1, k, "gamma1", 1)) - Phi_approx(get_base1(gamma1, (k - 1), "gamma1", 1))), 
                            "assigning variable lambda1");
                current_statement_begin__ = 29;
                stan::model::assign(lambda2, 
                            stan::model::cons_list(stan::model::index_uni(k), stan::model::nil_index_list()), 
                            (Phi_approx(get_base1(gamma2, k, "gamma2", 1)) - Phi_approx(get_base1(gamma2, (k - 1), "gamma2", 1))), 
                            "assigning variable lambda2");
            }
            current_statement_begin__ = 31;
            stan::model::assign(lambda1, 
                        stan::model::cons_list(stan::model::index_uni(K), stan::model::nil_index_list()), 
                        (1 - Phi_approx(get_base1(gamma1, (K - 1), "gamma1", 1))), 
                        "assigning variable lambda1");
            current_statement_begin__ = 32;
            stan::model::assign(lambda2, 
                        stan::model::cons_list(stan::model::index_uni(K), stan::model::nil_index_list()), 
                        (1 - Phi_approx(get_base1(gamma2, (K - 1), "gamma2", 1))), 
                        "assigning variable lambda2");
            if (!include_gqs__ && !include_tparams__) return;
            // validate transformed parameters
            const char* function__ = "validate transformed params";
            (void) function__;  // dummy to suppress unused var warning
            current_statement_begin__ = 15;
            stan::math::check_ordered(function__, "gamma1", gamma1);
            current_statement_begin__ = 16;
            stan::math::check_ordered(function__, "gamma2", gamma2);
            // write transformed parameters
            if (include_tparams__) {
                size_t gamma1_j_1_max__ = (K - 1);
                for (size_t j_1__ = 0; j_1__ < gamma1_j_1_max__; ++j_1__) {
                    vars__.push_back(gamma1(j_1__));
                }
                size_t gamma2_j_1_max__ = (K - 1);
                for (size_t j_1__ = 0; j_1__ < gamma2_j_1_max__; ++j_1__) {
                    vars__.push_back(gamma2(j_1__));
                }
                size_t lambda1_j_1_max__ = K;
                for (size_t j_1__ = 0; j_1__ < lambda1_j_1_max__; ++j_1__) {
                    vars__.push_back(lambda1(j_1__));
                }
                size_t lambda2_j_1_max__ = K;
                for (size_t j_1__ = 0; j_1__ < lambda2_j_1_max__; ++j_1__) {
                    vars__.push_back(lambda2(j_1__));
                }
            }
            if (!include_gqs__) return;
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e, current_statement_begin__, prog_reader__());
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }
    }
    template <typename RNG>
    void write_array(RNG& base_rng,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                     bool include_tparams = true,
                     bool include_gqs = true,
                     std::ostream* pstream = 0) const {
      std::vector<double> params_r_vec(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r_vec[i] = params_r(i);
      std::vector<double> vars_vec;
      std::vector<int> params_i_vec;
      write_array(base_rng, params_r_vec, params_i_vec, vars_vec, include_tparams, include_gqs, pstream);
      vars.resize(vars_vec.size());
      for (int i = 0; i < vars.size(); ++i)
        vars(i) = vars_vec[i];
    }
    std::string model_name() const {
        return "model_model_shift";
    }
    void constrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
        std::stringstream param_name_stream__;
        size_t alpha_j_1_max__ = (K - 1);
        for (size_t j_1__ = 0; j_1__ < alpha_j_1_max__; ++j_1__) {
            param_name_stream__.str(std::string());
            param_name_stream__ << "alpha" << '.' << j_1__ + 1;
            param_names__.push_back(param_name_stream__.str());
        }
        param_name_stream__.str(std::string());
        param_name_stream__ << "theta";
        param_names__.push_back(param_name_stream__.str());
        if (!include_gqs__ && !include_tparams__) return;
        if (include_tparams__) {
            size_t gamma1_j_1_max__ = (K - 1);
            for (size_t j_1__ = 0; j_1__ < gamma1_j_1_max__; ++j_1__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "gamma1" << '.' << j_1__ + 1;
                param_names__.push_back(param_name_stream__.str());
            }
            size_t gamma2_j_1_max__ = (K - 1);
            for (size_t j_1__ = 0; j_1__ < gamma2_j_1_max__; ++j_1__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "gamma2" << '.' << j_1__ + 1;
                param_names__.push_back(param_name_stream__.str());
            }
            size_t lambda1_j_1_max__ = K;
            for (size_t j_1__ = 0; j_1__ < lambda1_j_1_max__; ++j_1__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "lambda1" << '.' << j_1__ + 1;
                param_names__.push_back(param_name_stream__.str());
            }
            size_t lambda2_j_1_max__ = K;
            for (size_t j_1__ = 0; j_1__ < lambda2_j_1_max__; ++j_1__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "lambda2" << '.' << j_1__ + 1;
                param_names__.push_back(param_name_stream__.str());
            }
        }
        if (!include_gqs__) return;
    }
    void unconstrained_param_names(std::vector<std::string>& param_names__,
                                   bool include_tparams__ = true,
                                   bool include_gqs__ = true) const {
        std::stringstream param_name_stream__;
        size_t alpha_j_1_max__ = (K - 1);
        for (size_t j_1__ = 0; j_1__ < alpha_j_1_max__; ++j_1__) {
            param_name_stream__.str(std::string());
            param_name_stream__ << "alpha" << '.' << j_1__ + 1;
            param_names__.push_back(param_name_stream__.str());
        }
        param_name_stream__.str(std::string());
        param_name_stream__ << "theta";
        param_names__.push_back(param_name_stream__.str());
        if (!include_gqs__ && !include_tparams__) return;
        if (include_tparams__) {
            size_t gamma1_j_1_max__ = (K - 1);
            for (size_t j_1__ = 0; j_1__ < gamma1_j_1_max__; ++j_1__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "gamma1" << '.' << j_1__ + 1;
                param_names__.push_back(param_name_stream__.str());
            }
            size_t gamma2_j_1_max__ = (K - 1);
            for (size_t j_1__ = 0; j_1__ < gamma2_j_1_max__; ++j_1__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "gamma2" << '.' << j_1__ + 1;
                param_names__.push_back(param_name_stream__.str());
            }
            size_t lambda1_j_1_max__ = K;
            for (size_t j_1__ = 0; j_1__ < lambda1_j_1_max__; ++j_1__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "lambda1" << '.' << j_1__ + 1;
                param_names__.push_back(param_name_stream__.str());
            }
            size_t lambda2_j_1_max__ = K;
            for (size_t j_1__ = 0; j_1__ < lambda2_j_1_max__; ++j_1__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "lambda2" << '.' << j_1__ + 1;
                param_names__.push_back(param_name_stream__.str());
            }
        }
        if (!include_gqs__) return;
    }
}; // model
}  // namespace
typedef model_model_shift_namespace::model_model_shift stan_model;
#ifndef USING_R
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}
#endif
#endif
