#ifndef AttentionNet_H
#define AttentionNet_H

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
class AttentionNet : torch::nn::Module {
public:
    AttentionNet() {};
    

private:
};

#endif
