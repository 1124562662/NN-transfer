#ifndef AttentionNet_H
#define AttentionNet_H

#include "Variables.h"
#include "dependency.h"
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

class AttentionNet : torch::nn::Module {
public:
    AttentionNet(){};
    AttentionNet(int attention_layer_output_size, bool ReadModulesParams,
                 string ReadModulesParams_Path, bool remove_file,
                 map<string, int> stateVariableIndices_net,
                 map<string, int> action_indices_)
        : stateVariableIndices_net(stateVariableIndices_net),
          action_indices_(action_indices_),
          attention_layer_output_size(attention_layer_output_size),
          ReadModulesParams(ReadModulesParams),
          ReadModulesParams_Path(ReadModulesParams_Path),
          remove_file(remove_file) {
        this->initialize(this->filename);

        if (!remove_file) {
            this->save_indices();
        };
    };

    // training only
    AttentionNet(string train_folder, int attention_layer_output_size,
                 bool ReadModulesParams, string ReadModulesParams_Path,
                 bool remove_file)
        : attention_layer_output_size(attention_layer_output_size),
          ReadModulesParams(ReadModulesParams),
          remove_file(remove_file) {
        this->read_indices(train_folder);
        this->ReadModulesParams_Path = train_folder + ReadModulesParams_Path;
        this->initialize(train_folder + (this->filename));
    };

    void save_parameters(string path);

    static void trainnet_();

    void set_modules(
        map<string, nn_module>& action_modules2,
        map<string, map<string, nn_module>>& fluents_descriptors2,
        map<string, nn_module>& action_modules_attentions2,
        map<string, map<string, nn_module>>& aggregation_attentions2);

    torch::Tensor forward(vector<double> const& current_state_);
    torch::Tensor forward();
    void set_state_pointer(vector<double>* stateVec) {
        this->state_net = stateVec;
    };

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
        map<string, int>& action_modules_sizes1,
        map<string, map<string, int>>& fluent_descriptors_sizes1,

        set<AbstractVariable, classcomp> const& StateFluentsSchema2,
        set<AbstractVariable, classcomp> const& NonFluentsSchema2,
        set<AbstractVariable, classcomp> const& ActionFluentsSchema2,
        map<string, int>& action_modules_sizes2,
        map<string, map<string, int>>& fluent_descriptors_sizes2);

    // ------(2)instance------
    vector<InstantiatedVariable> StateFluents;
    vector<InstantiatedVariable> NonFluents;
    vector<InstantiatedVariable> ActionFluents;

    // ------(3) dependency information------

    // fullname of action -> their dependencies
    map<string, Action_module_dependency> action_modules_dependencies;

    // action abstract name -> the input size of the module
    map<string, int> action_modules_sizes;

    // fullname of action -> the restricted fluents' IDs-> Instantiated
    // fluents's fullnames ==> Fluent_descriptor_dependency
    map<string, map<string, map<string, Fluent_descriptor_dependency>>>
        fluent_descriptors_dependencies;

    //  action type -> the restricted fluents' IDs -> the input size of
    // the fluent_descriptor
    map<string, map<string, int>> fluent_descriptors_sizes;

    // ------(4) modules ------

    // action name -> modules
    map<string, nn_module> action_modules;
    // action type => ID of the restricted fluents' IDs   => fluent
    // descriptor modules
    map<string, map<string, nn_module>> fluents_descriptors;

    // ------(5) attention modules ------

    // action type -> attention values over ID of the restricted fluents'
    // types
    map<string, nn_module> action_modules_attentions;

    //  action type -> ID of the restricted fluents' IDs  -> attention
    // values over outputs of fluents' descriptors
    map<string, map<string, nn_module>> aggregation_attentions;

    // ------(6) record index ------
    map<string, int> stateVariableIndices_net;
    map<string, int> action_indices_;
    vector<string> res_order;

    void change_instance(AttentionNet& a2);
    void save_indices();
    void read_indices(string path);
    vector<double> tensor2vector_z(torch::Tensor in);

private:
    int attention_layer_output_size;
    bool ReadModulesParams;
    string ReadModulesParams_Path;
    string filename = "Net2_information";
    bool remove_file;

    vector<double>* state_net;

    void read_parameters();

    void initialize(string filepath);

    void initialize_readHelper(string filepath);
    void initialize_ActionM_input_Helper(
        InstantiatedVariable& mainAction,
        set<AbstractVariable, classcomp> const& _FluentsSchema,
        vector<AbstractRestrictedVariable>& dependent_abs_fluents,
        vector<int>& Inputs_types);
    static vector<double> tensor2vector(torch::Tensor in);
    static bool vec_is_intersect(vector<string> v1, vector<string> v2);
    static bool v2_contains_v1(vector<string> v1, vector<string> v2);
    static bool v1_contains_diff_from_v2(vector<string> v1, vector<string> v2);
    void initialize_contains_helper(
        const set<AbstractVariable, classcomp>::iterator it,
        InstantiatedVariable& mainAction, int aggregate_type,
        set<string>& test_duplicated, vector<int>& Inputs_types,
        vector<AbstractRestrictedVariable>& dependent_abs_fluents);
    void initialize_Instantiation_helper(
        vector<InstantiatedVariable>& fluents_,
        AbstractRestrictedVariable& tmpARVar,
        vector<InstantiatedVariable>& Instantiated_ones);
    Fluent_descriptor_dependency initialize_fluent_descriptor_helper(
        InstantiatedVariable& actionF, AbstractRestrictedVariable& Rfluent,
        InstantiatedVariable& Ifluent);
    void initialize_typeMatching_helper(
        int const& param_index, int aggregate_type, vector<int>& tmp1,
        vector<string>& tmp2, vector<int>& tmp3, vector<string>& tmp4,
        vector<string>& ResVarIDs,
        vector<vector<int>>& ObjectsRestriction_loc_s,
        vector<vector<string>>& ObjectsRestriction_s);
    void initialize_Instantiation_(
        vector<AbstractRestrictedVariable>& dependent_abs_fluents,
        map<string, vector<InstantiatedVariable>>& dependent_ins_fluents);
    void set_action_modules_size(string aname, int size);
    void set_fluent_descriptors_size(string action_name, string ResVarID,
                                     int size);
    void initialize_modules();

    void initialize_modules_helper();
    static bool v_unique(const vector<string>& vec);
};

#endif
