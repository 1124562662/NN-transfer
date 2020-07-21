#include "nn_module.h"
#include <torch/torch.h>

#include "assert.h"
#include <algorithm>
#include <iostream>

using namespace std;
nn_module::nn_module(string name, vector<int>& topology)
    : name(name), topology(topology) {
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);
    for (unsigned int i = 0; i < topology.size() - 1; ++i) {
        torch::Tensor tmp_weights =
            torch::empty({topology[i], topology[i + 1]}, options);
        tmp_weights = torch::nn::init::kaiming_normal_(
            tmp_weights, 0.01, torch::kFanOut, torch::kLeakyReLU);

        tmp_weights.requires_grad_(true);
        tmp_weights = register_parameter(name + ",weight:" + std::to_string(i),
                                         tmp_weights);
        this->layer_weights_sub.push_back(tmp_weights);

        torch::Tensor tmp_bias = torch::randn({topology[i + 1]}, options);
        tmp_bias =
            register_parameter(name + ",bias:" + std::to_string(i), tmp_bias);
        this->layer_biases_sub.push_back(tmp_bias);
    }
}

void nn_module::save_submodel(string path, string nn_module_name) {
    std::ofstream taskFile;
    string taskFileName = path + nn_module_name;
    taskFile.open(taskFileName.c_str());
    taskFile << "name" << endl;
    taskFile << this->name << endl;
    taskFile << "topologySize" << endl;
    taskFile << this->topology.size() << endl;
    taskFile.close();
    // save params
    for (int i = 0; i < this->layer_weights_sub.size(); i++) {
        torch::save((this->layer_weights_sub)[i],
                    path + name + ",weight:" + std::to_string(i));
    }
    for (int i = 0; i < this->layer_biases_sub.size(); i++) {
        torch::save((this->layer_biases_sub)[i],
                    path + name + ",bias:" + std::to_string(i));
    }
}

void nn_module::Load_submodel(string path, string nn_module_name) {
    int topologySize;
    std::ifstream taskFile;
    string taskFileName = path + nn_module_name;
    taskFile.open(taskFileName.c_str());
    if (taskFile.is_open()) {
        std::string line;
        getline(taskFile, line);
        assert(line == "name");
        getline(taskFile, line);
        this->name = line;
        getline(taskFile, line);
        assert(line == "topologySize");
        topologySize = stoi(line);
        cout << "should be int" << to_string(topologySize) << endl;
        taskFile.close();
    }
    // read Params
    assert(this->layer_weights_sub.size() == 0);
    assert(this->layer_biases_sub.size() == 0);
    for (int i = 0; i < topologySize - 1; i++) {
        torch::Tensor weights;
        torch::load(weights, path + name + ",weight:" + std::to_string(i));
        weights = weights.requires_grad_(true);
        weights =
            register_parameter(name + ",weight:" + std::to_string(i), weights);

        this->layer_weights_sub.push_back(weights);

        torch::Tensor bias;
        torch::load(bias, path + name + ",bias:" + std::to_string(i));
        bias = bias.requires_grad_(true);
        bias = register_parameter(name + ",bias:" + std::to_string(i), bias);

        this->layer_biases_sub.push_back(bias);
    }
}

torch::Tensor nn_module::forward(torch::Tensor X_input) {
    for (unsigned int i = 0; i < this->layer_weights_sub.size(); ++i) {
        X_input =
            torch::leaky_relu(torch::addmm(this->layer_biases_sub[i], X_input,
                                           (this->layer_weights_sub)[i]),
                              0.01);
    }
    return X_input;
}