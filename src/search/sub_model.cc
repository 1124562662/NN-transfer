#include "sub_model.h"
#include <torch/torch.h>

#include <algorithm>
#include <iostream>

using namespace std;

void sub_model::save_submodel(std::string path) {
    for (int i = 0; i < this->layer_weights_sub.size(); i++) {
        cout << path + "layer_weights_sub" + std::to_string(i) << endl;
        torch::save((this->layer_weights_sub)[i],
                    path + "layer_weights_sub" + std::to_string(i));
    }
    for (int i = 0; i < this->layer_biases_sub.size(); i++) {
        torch::save((this->layer_biases_sub)[i],
                    path + "layer_bias_sub" + std::to_string(i));
    }
}
void sub_model::initialize(std::string path, int layer_size_) {
    assert(layer_size_ = this->sub_mod_size - 1);
    for (int i = 0; i < layer_size_; i++) {
        torch::Tensor weights;
        torch::load(weights, path + "layer_weights_sub" + std::to_string(i));
        weights = weights.requires_grad_(true);
        weights = register_parameter("weights" + std::to_string(i), weights);
        this->layer_weights_sub.push_back(weights);

        torch::Tensor bias;
        torch::load(bias, path + "layer_bias_sub" + std::to_string(i));
        bias = bias.requires_grad_(true);
        bias = register_parameter("bias" + std::to_string(i), bias);
       
        this->layer_biases_sub.push_back(bias);
    }
}

// if not provided, random initiliazation
void sub_model::initialize(std::vector<int> topology) {
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);
    assert(topology.size() == this->sub_mod_size);
    for (unsigned int i = 0; i < topology.size() - 1; ++i) {
        torch::Tensor tmp_weights =
            torch::empty({topology[i], topology[i + 1]}, options);
        tmp_weights = torch::nn::init::kaiming_normal_(
            tmp_weights, 0.01, torch::kFanOut, torch::kLeakyReLU);
        tmp_weights.requires_grad_(true);
        tmp_weights =
            register_parameter("weight_" + std::to_string(i), tmp_weights);
        this->layer_weights_sub.push_back(tmp_weights);

        torch::Tensor tmp_bias = torch::randn({topology[i + 1]}, options);
        tmp_bias = register_parameter("bias_" + std::to_string(i), tmp_bias);
        this->layer_biases_sub.push_back(tmp_bias);
    }
    // this->fc1 =
    //     register_module("fc1", torch::nn::Linear(topology[0], topology[1]));
    // this->fc2 =
    //     register_module("fc2", torch::nn::Linear(topology[1], topology[2]));
}

torch::Tensor sub_model::forward(torch::Tensor X_input) {
    for (unsigned int i = 0; i < this->layer_weights_sub.size(); ++i) {
        X_input = torch::leaky_relu(torch::addmm(
            this->layer_biases_sub[i], X_input, (this->layer_weights_sub)[i]),0.01);
    }
//cout<<"ll"<<X_input.sizes()<<","<<to_string(this->layer_weights_sub.size())<< endl;
    // X_input = torch::relu(fc1->forward(X_input));
    // X_input = torch::relu(fc2->forward(X_input));
    return X_input;
}