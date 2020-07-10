#ifndef Net_H
#define Net_H

#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <torch/torch.h>

#include "dependency_info.h"
#include "sub_model.h"

#include <cassert>
#include <iostream>
#include <set>
// Evaluates all actions by simulating a run that starts with that action
// followed by a  NN until a terminal state is reached
using namespace std;
class Net : torch::nn::Module {
public:
    Net() {
        value_set = false;
    }

    struct Nett : torch::nn::Module {
        Nett() {
            // Construct and register two Linear submodules.
            fc1 = register_module("fc1", torch::nn::Linear(10, 7));
            fc2 = register_module("fc2", torch::nn::Linear(7, 6));
            fc3 = register_module("fc3", torch::nn::Linear(6, 5));
        }

        // Implement the Net's algorithm.
        torch::Tensor forward(std::vector<double> xs) {
            auto options = torch::TensorOptions().requires_grad(true);
            torch::Tensor x = torch::full({1, 1}, 0, options);
            for (int i = 0; i < xs.size(); i++) {
                x = torch::cat({x, torch::full({1, 1}, (float)xs[i], options)},
                               1);
            }
            // Use one of many tensor manipulation functions.
            x = torch::relu(fc1->forward(x));
            x = torch::relu(fc2->forward(x));
            x = fc3->forward(x);
            return x;
        }

        // Use one of many "standard library" modules.
        torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    };

    // only use the first 2 parameters if read from file
    void initialize_(bool read_parameters, std::string filename_,
                     int output_size_subM);
    void initialize_(
        bool read_parameters, std::string filename_, int output_size_subM,
        std::map<std::string, double>* NonfluentsToValues_net,
        std::map<std::string, std::map<std::string, std::set<std::string>>>*
            sf2VariedSize_sfs_net,
        std::map<std::string, std::set<std::string>>* sf2FixedSize_sfs_net,
        std::map<std::string, int>* stateVariableIndices_net,
        std::vector<std::vector<std::string>>* stateVariableValues_net,
        std::vector<std::string> action_f_net,
        std::map<std::string, int>* action_indices_

    );

    torch::Tensor forward(std::vector<double> const& current_state_);
    // used for training
    torch::Tensor forward(
        std::map<std::string, torch::Tensor> const& layer_Minus1);
    torch::Tensor forward();

    std::vector<long double> tensor2vector_z(torch::Tensor in);

    void save_parameters(std::string file_name);

    static void trainnet_();

    // instantiated state fluent-> its dependency_info
    std::map<std::string, dependency_info> sub_models_dependency_information_;

    std::map<int /*layer No.*/, std::map<std::string /*sf type*/, sub_model>>
        torch_sub_models;

    // key : string of state fluents with parameters
    // value : a vector of fixed-size dependent state fluents with
    // parametetrs
    std::map<std::string, std::set<std::string>>* sf2FixedSize_sfs_net;

    // key : string of state fluents with parameters -> quantifiers
    // -> vector of parameters and dependent state fluents
    std::map<std::string, std::map<std::string, std::set<std::string>>>*
        sf2VariedSize_sfs_net;

    std::map<std::string, double>* NonfluentsToValues_net;

    // pointer to the ones in ipc_client
    std::map<std::string, int>* stateVariableIndices_net;
    std::vector<std::vector<std::string>>* stateVariableValues_net;

    // only used if input of forward is not provieded.
    std::vector<double>* state_net;

    // instantiated actions
    std::vector<std::string> action_f_net;

    std::map<std::string, int>* action_indices_;

    // layer of the network.
    int layer = 3;

    int output_size_subM;

    std::set<std::string> Nonfluents_types;

    void set_state_pointer(std::vector<double>* stateVec) {
        this->state_net = stateVec;
        this->state_pointer_set = true;
    };

private:
    torch::Tensor forward_helper();
    bool state_pointer_set = false;
    std::vector<std::string> res_order;
    void get_remaining(
        std::map<std::string, std::set<std::string>>& remaining_SFs,
        int& input_size_subm, int& input_size_subm_zero,
        std::set<std::string>& used);

    std::map<int, std::map<std::string, torch::Tensor>> layer_output;

    static std::string erasechar(std::string str);

    std::vector<std::string> get_param_vec(std::string str);

    std::string get_variable_name(std::string str);

    void build_input_helper(bool& set, std::string& s1, torch::Tensor& in1,
                            int i, std::string pooling_type);

    bool check_layer_output_exist(std::string s1, int layer_num);

    bool check_if_is_AF(std::string str); // check if str is an action fluent
    void add_input_size_subm(std::string name, int& input_size_subm);
    void initialize_action_subMod(
        int submodule_size); // only called by initialize_()

    void initialize_sf_subMod(
        int submodule_size); // only called by initialize_()

    std::vector<std::string>
    initialize_nonfluent_Noparams(); // only called by initialize_()

    std::vector<std::string> get_abs_vec_of_dependent_varied(
        std::string instantiated_dependent_sf, std::string main_sf,
        std::string quantifier); // used when instantiating

    bool vec_is_intersect(std::vector<std::string> v1,
                          std::vector<std::string> v2);
    bool get_instantieted_ones(
        std::vector<std::string>& params_abstract,
        std::vector<std::string>& params_instance,
        std::vector<std::string>& parameters_dependt_abst,
        std::vector<std::string>& parameters_dependt_inst);

    void get_tmp_map(std::map<std::string, std::vector<std::string>>* in);
    bool value_set = false;
};

#endif
