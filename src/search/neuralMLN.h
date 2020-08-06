#ifndef NEURALMLN_H
#define NEURALMLN_H

#include "Variables.h"
#include "dependency.h"
#include "mln.h"
#include "nn_module.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <cassert>
#include <iostream>
#include <set>
#include <torch/torch.h>
using namespace std;
class AbstractVariable;
class InstantiatedVariable;
class Action_module;

class neuralMLN : torch::nn::Module {
public:
    neuralMLN(){};
    neuralMLN(int attention_layer_output_size, bool ReadModulesParams,
              string ReadModulesParams_Path, bool remove_file,
              /* map<string, int> stateVariableIndices_net,
               map<string, int> action_indices_, */
              int filter_size)
        : /*  stateVariableIndices_net(stateVariableIndices_net),
       action_indices_(action_indices_),*/
          attention_layer_output_size(attention_layer_output_size),
          ReadModulesParams(ReadModulesParams),
          ReadModulesParams_Path(ReadModulesParams_Path),
          remove_file(remove_file),
          filter_size(filter_size) {
        this->initialize(this->filename);
    };

    // training only
    neuralMLN(string train_folder, int attention_layer_output_size,
              bool ReadModulesParams, string ReadModulesParams_Path,
              bool remove_file)
        : attention_layer_output_size(attention_layer_output_size),
          ReadModulesParams(ReadModulesParams),
          remove_file(remove_file) {
        this->read_indices(train_folder);
        this->ReadModulesParams_Path = train_folder + ReadModulesParams_Path;
    };

    void save_parameters(string path);

    static void trainnet_();

    torch::Tensor forward(vector<double> const& current_state_);

    //------(1)domian schema------
    struct classcomp {
        bool operator()(const AbstractVariable& lhs,
                        const AbstractVariable& rhs) const {
            return lhs.name < rhs.name;
        }
    };
    set<AbstractVariable, classcomp> StateFluentsSchema;
    set<AbstractVariable, classcomp> NonFluentsSchema;
    set<AbstractVariable, classcomp> ActionFluentsSchema;

    // order matters !
    static bool check_domain_schema(
        set<AbstractVariable, classcomp> const& StateFluentsSchema1,
        set<AbstractVariable, classcomp> const& NonFluentsSchema1,
        set<AbstractVariable, classcomp> const& ActionFluentsSchema1,
        set<AbstractVariable, classcomp> const& StateFluentsSchema2,
        set<AbstractVariable, classcomp> const& NonFluentsSchema2,
        set<AbstractVariable, classcomp> const& ActionFluentsSchema2);

    // ------(2)instance------
    vector<InstantiatedVariable> StateFluents;
    vector<InstantiatedVariable> NonFluents;
    vector<InstantiatedVariable> ActionFluents;

    // ------(3) record index ------
    map<string, int> stateVariableIndices_net;
    map<string, int> action_indices_;

    void change_instance(neuralMLN& a2);
    void save_indices();
    void read_indices(string path);

    // -----------------(4)  MLN----------------

    myMLN mln_;

    map<int, filter> Fliters;
    map<int, vector<vector<InstantiatedVariable>>> InstantiatedInputs;

    // ID of filter Nets -> its aggregation attention module
    map<int, nn_module> FilterNets;
    map<int, nn_module> Aggregate_attentions;
    

    void save_filters();
    void read_filters(string path);
    void initialize(string file);
   

private:
    int attention_layer_output_size;
    bool ReadModulesParams;
    string ReadModulesParams_Path;
    string filename = "Net2_information";
    bool remove_file;
    int filter_size;

    void ins_helper(
        set<string>& duplictes, vector<AbstractVariable> remaining,
        vector<InstantiatedVariable> Instantiated_ones,
        vector<vector<pair<pair<int, int>, pair<int, int>>>>& restriction_s,
        vector<vector<InstantiatedVariable>>& result);

    void initialize_readHelper1(string filepath);

    vector<InstantiatedVariable> find_ins(
        AbstractVariable a,
        vector<vector<pair<pair<int, int>, pair<int, int>>>>& restriction_s,
        vector<InstantiatedVariable>& Instantiated_ones, int idx);

    static bool vec_is_intersect(vector<string> v1, vector<string> v2);

    static bool v2_contains_v1(vector<string> v1, vector<string> v2);

    static bool v1_contains_diff_from_v2(vector<string> v1, vector<string> v2);
};

#endif
