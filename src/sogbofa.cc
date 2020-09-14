#include "sogbofa.h"
#include "assert.h"
#include "utils/system_utils.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <torch/torch.h>

using namespace std;

// this->sogbofa_parser.reward_function.print();
// cout << endl;
// for (size_t i = 0; i < sogbofa_parser.CPFs.size(); i++)
// {
//     sogbofa_parser.CPFs[i].print();
//     cout<<endl;
// }

torch::Tensor sogbofa::guided_aggregated_seach(
    map<string, torch::Tensor>& current_state, int horizon) {
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    torch::Tensor res = torch::full({1, 1}, 0.0, options);

    // set up the r net Input
    rn->set_instance_tensor();

    for (size_t hrz = 0; hrz < horizon; hrz++) {
        map<string, torch::Tensor> action_map =
            this->rn->forward_sogbofa(current_state);
        res = torch::add(res,
                         this->reward_for_one_state(current_state, action_map));

        map<string, torch::Tensor> successor_belif_state;
        this->compute_successor_belif_state(current_state, action_map,
                                            successor_belif_state);
        current_state = successor_belif_state;
    }
    return res;
}

void sogbofa::compute_successor_belif_state(
    map<string, torch::Tensor>& state_fluent_input,
    map<string, torch::Tensor>& action_fluent_input,
    map<string, torch::Tensor>& successor_belif_state) {
    map<int, string>::iterator it;
    for (it = this->sogbofa_parser.statefluents_indices.begin();
         it != this->sogbofa_parser.statefluents_indices.end(); it++) {
        string sfname = it->second;
        sogbofa_expr* SSA = &(this->sogbofa_parser.CPFs[it->first]);
        successor_belif_state[sfname] =
            SSA->evaluate(state_fluent_input, action_fluent_input);
    };
}

torch::Tensor sogbofa::reward_for_one_state(
    map<string, torch::Tensor>& state_fluent_input,
    map<string, torch::Tensor>& action_fluent_input) {
   
    sogbofa_expr* r_expr = &(this->sogbofa_parser.reward_function);
    torch::Tensor res_reward =
        r_expr->evaluate(state_fluent_input, action_fluent_input);
    return res_reward;
}