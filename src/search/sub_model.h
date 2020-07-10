#ifndef sub_model_H
#define sub_model_H

#include <torch/torch.h>

class sub_model : public torch::nn::Module {
public:
   sub_model() {
      std::cout<<"Defalut constructor of submodel used."<<std::endl;
    };
    sub_model(int sub_mod_size) {
        this->sub_mod_size = sub_mod_size;
        std::cout<<"Int   constructor of submodel used."<<std::endl;
    };
    int sub_mod_size;

    void initialize(std::vector<int> topology);
    //  X_input must be a matrix
    void initialize(std::string filepath, int layer_size_);

    torch::Tensor forward(torch::Tensor X_input);
    void save_submodel(std::string path);

    std::vector<torch::Tensor> layer_weights_sub;
    std::vector<torch::Tensor> layer_biases_sub;
};
#endif
