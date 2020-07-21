#include "AttentionNet.h"
#include <torch/torch.h>

#include "utils/math_utils.h"

#include "Variables.h"
#include "dependency.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
using namespace std;

torch::Tensor AttentionNet::forward(vector<double> const& current_state_) {
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    // fullname of fluents -> their values
    map<string, torch::Tensor> Inputs;
    for (int i = 0; i < (this->StateFluents).size(); i++) {
        torch::Tensor tensor_ =
            torch::full({1, 1}, (this->StateFluents)[i].value, options);
        Inputs[(this->StateFluents)[i].fullname] = tensor_;
    }

    std::map<std::string, int>::iterator it;
    for (it = this->stateVariableIndices_net.begin();
         it != this->stateVariableIndices_net.end(); it++) {
        torch::Tensor tensor_ =
            torch::full({1, 1}, current_state_[it->second], options);
        Inputs[it->first] = tensor_;
    };
    for (int i = 0; i < this->NonFluents.size(); i++) {
        torch::Tensor tensor_ =
            torch::full({1, 1}, (this->NonFluents)[i].value, options);
        Inputs[(this->NonFluents)[i].fullname] = tensor_;
    }

    // fullname of the action -> result of the action module
    map<string, torch::Tensor> ResMap;

    for (int i = 0; i < this->ActionFluents.size(); i++) {
        string aFullName = (this->ActionFluents)[i].fullname;
        string aName = (this->ActionFluents)[i].name;

        Action_module_dependency a_module_dependency =
            (this->action_modules_dependencies)[aFullName];

        // absVar.ID -> type keys
        map<string, torch::Tensor> TYPE_keys;
        map<string, torch::Tensor> Vs;

        for (int j = 0; j < a_module_dependency.dependent_abs_fluents.size();
             j++) {
            AbstractRestrictedVariable absVar =
                a_module_dependency.dependent_abs_fluents[j];

            if (a_module_dependency.Inputs_types[j] == 2) {
                nn_module* fluents_descriptor =
                    &((this->fluents_descriptors)[aName][absVar.ID]);
                nn_module* aggregat_attention_m =
                    &((this->aggregation_attentions)[aName][absVar.ID]);

                // insVar.fullname -> key
                map<string, torch::Tensor> keys;
                map<string, torch::Tensor> attentionValues;
                torch::Tensor quotient = torch::full({1, 1}, 0.0, options);

                for (int k = 0;
                     k < a_module_dependency.dependent_ins_fluents[absVar.ID]
                             .size();
                     k++) {
                    InstantiatedVariable insVar =
                        a_module_dependency.dependent_ins_fluents[absVar.ID][k];
                    Fluent_descriptor_dependency f_descriptor_dependency =
                        (this->fluent_descriptors_dependencies)
                            [aFullName][absVar.ID][insVar.fullname];
                    torch::Tensor tensor_;
                    for (int q = 0;
                         q < f_descriptor_dependency
                                 .descriptor_dependent_abs_fluents.size();
                         q++) {
                        AbstractRestrictedVariable absVar2 =
                            f_descriptor_dependency
                                .descriptor_dependent_abs_fluents[q];

                        vector<InstantiatedVariable> Inst_ones =
                            f_descriptor_dependency
                                .descriptor_dependent_ins_fluents[absVar2.ID];
                        torch::Tensor tensor_2;
                        for (int q2 = 0; q2 < Inst_ones.size(); q2++) {
                            torch::Tensor d = Inputs[Inst_ones[q2].fullname];
                            if (q2 == 0) {
                                tensor_2 = d;
                            } else {
                                tensor_2 = torch::max(tensor_2, d);
                            }
                        }
                        if (tensor_2.size(1) == 0) {
                            cout << "no instantiation?" << endl;
                            tensor_2 = torch::full({1, 1}, 0.0, options);
                        };

                        if (q == 0) {
                            tensor_ = tensor_2;
                        } else {
                            tensor_ = torch::cat({tensor_, tensor_2}, 1);
                        };
                    };

                    torch::Tensor out1 = fluents_descriptor->forward(tensor_);
                    keys[insVar.fullname] = out1;

                    // attention 1
                    torch::Tensor a1 =
                        torch::exp(aggregat_attention_m->forward(out1));

                    attentionValues[insVar.fullname] = a1;

                    quotient = torch::add(quotient, a1);
                }

                map<string, torch::Tensor> attentionValues2;
                map<string, torch::Tensor>::iterator it2;
                for (it2 = attentionValues.begin();
                     it2 != attentionValues.end(); it2++) {
                    attentionValues2[it2->first] =
                        torch::div(it2->second, quotient);
                }

                torch::Tensor aggregated_keys;
                torch::Tensor aggregated_vs = torch::full({1, 1}, 0.0, options);

                // aggregation
                for (int k = 0;
                     k < a_module_dependency.dependent_ins_fluents[absVar.ID]
                             .size();
                     k++) {
                    InstantiatedVariable insVar =
                        a_module_dependency.dependent_ins_fluents[absVar.ID][k];
                    torch::Tensor tmp =
                        torch::mul(attentionValues2[insVar.fullname],
                                   Inputs[insVar.fullname]);
                    aggregated_vs = torch::add(aggregated_vs, tmp);

                    torch::Tensor tmp2 =
                        torch::mul(attentionValues2[insVar.fullname],
                                   keys[insVar.fullname]);

                    if (k == 0) {
                        aggregated_keys = tmp2;
                    } else {
                        aggregated_keys = torch::add(aggregated_keys, tmp2);
                    };
                };
                TYPE_keys[absVar.ID] = aggregated_keys;
                Vs[absVar.ID] = aggregated_vs;

            } else {
                // No aggregation nedded
                vector<InstantiatedVariable> InstOnes =
                    a_module_dependency.dependent_ins_fluents[absVar.ID];
                torch::Tensor t1;
                if (InstOnes.size() == 1) {
                    t1 = Inputs[InstOnes[0].fullname];
                } else if (InstOnes.size() == 0) {
                    t1 = torch::full({1, 1}, 0.0, options);
                    cout << "0 instantiation encountered." << endl;
                } else {
                    SystemUtils::abort("Size larger than 1." +
                                       to_string(InstOnes.size()));
                }
                Vs[absVar.ID] = t1;
            }
        };

        // attention 2
        map<string, torch::Tensor>::iterator it_;
        nn_module* a_modules_attention =
            &(this->action_modules_attentions[aName]);

        for (it_ = TYPE_keys.begin(); it_ != TYPE_keys.end(); it_++) {
            torch::Tensor atv =
                torch::sigmoid(a_modules_attention->forward(it_->second));
            torch::Tensor newV = torch::mul(atv, Vs[it_->first]);

            Vs[it_->first] = newV;
        }
        torch::Tensor ActionInput;
        int idx = 0;
        for (it_ = Vs.begin(); it_ != Vs.end(); it_++) {
            if (idx == 0) {
                ActionInput = it_->second;
            } else {
                ActionInput = torch::cat({ActionInput, it_->second}, 1);
            };
            idx += 1;
        }
        torch::Tensor res_ = (this->action_modules)[aName].forward(ActionInput);
        ResMap[aFullName] = res_;
    };

    torch::Tensor res = ResMap["no-op"];
    for (int i = 1; i < this->res_order.size(); i++) {
        res = torch::cat({res, ResMap[(this->res_order)[i]]}, 1);
    };
    return res;
}

void AttentionNet::set_modules(
    map<string, nn_module>& action_modules2,
    map<string, map<string, nn_module>>& fluents_descriptors2,
    map<string, nn_module>& action_modules_attentions2,
    map<string, map<string, nn_module>>& aggregation_attentions2) {
    this->action_modules = action_modules2;
    this->fluents_descriptors = fluents_descriptors2;
    this->action_modules_attentions = action_modules_attentions2;
    this->aggregation_attentions = aggregation_attentions2;
}

void AttentionNet::initialize(string filepath) {
    this->initialize_readHelper(filepath);
    // helper structure goes here

    for (int i = 0; i < this->ActionFluents.size(); i++) {
        // (a) action dependecy part
        vector<AbstractRestrictedVariable> dependent_abs_fluents;
        // 1 ->  no aggregation needed
        // 2 -> aggregation needed
        vector<int> Inputs_types;
        this->initialize_ActionM_input_Helper(
            this->ActionFluents[i], this->NonFluentsSchema,
            dependent_abs_fluents, Inputs_types);

        this->initialize_ActionM_input_Helper(
            this->ActionFluents[i], this->StateFluentsSchema,
            dependent_abs_fluents, Inputs_types);

        // instantiation
        map<string, vector<InstantiatedVariable>> dependent_ins_fluents;
        this->initialize_Instantiation_(dependent_abs_fluents,
                                        dependent_ins_fluents);
        //

        Action_module_dependency action_module_dependency_ =
            Action_module_dependency(this->ActionFluents[i],
                                     dependent_abs_fluents, Inputs_types,
                                     dependent_ins_fluents);

        this->set_action_modules_size(this->ActionFluents[i].name,
                                      dependent_abs_fluents.size());

        (this->action_modules_dependencies)[ActionFluents[i].fullname] =
            action_module_dependency_;
        // (b) fluent descriptor dependecy part
        map<string, map<string, Fluent_descriptor_dependency>> map1;
        for (int j = 0; j < dependent_abs_fluents.size(); j++) {
            vector<InstantiatedVariable> Inst_ones =
                dependent_ins_fluents[dependent_abs_fluents[j].ID];
            map<string, Fluent_descriptor_dependency> map2;
            for (int jj = 0; jj < Inst_ones.size(); jj++) {
                map2[Inst_ones[jj].fullname] =
                    this->initialize_fluent_descriptor_helper(
                        this->ActionFluents[i], dependent_abs_fluents[j],
                        Inst_ones[jj]);
            };
            map1[(dependent_abs_fluents[j]).ID] = map2;
        };
        (this->fluent_descriptors_dependencies)[this->ActionFluents[i]
                                                    .fullname] = map1;
    };

    // module part
    this->initialize_modules();

    // handle the indices
    std::vector<std::string> tmpv(this->action_indices_.size() + 1);
    this->res_order = tmpv;
    std::map<std::string, int>::iterator itt;
    for (itt = this->action_indices_.begin();
         itt != this->action_indices_.end(); itt++) {
        (this->res_order)[itt->second] = itt->first;
    };

    for (int i = 0; i < this->ActionFluents.size(); i++) {
        if ((this->ActionFluents)[i].fullname != "no-op") {
            bool found = false;
            for (int j = 0; j < (this->res_order).size(); j++) {
                if ((this->res_order)[j] == (this->ActionFluents)[i].fullname) {
                    found = true;
                };
            };
            if (!found) {
                SystemUtils::abort(
                    " (this->ActionFluents)[i].fullname  NOT IN res_order");
            };
        };
    };
    cout << "=======================" << endl;
}

void AttentionNet::initialize_ActionM_input_Helper(
    InstantiatedVariable& mainAction,
    set<AbstractVariable, classcomp> const& _FluentsSchema,
    vector<AbstractRestrictedVariable>& dependent_abs_fluents,
    vector<int>& Inputs_types) {
    set<string> test_duplicated;

    set<AbstractVariable, classcomp>::iterator it;

    for (it = _FluentsSchema.begin(); it != _FluentsSchema.end(); it++) {
        int aggregate_type;
        AbstractRestrictedVariable absRvar;

        if (it->patameters_types.size() == 0) {
            // 1 -> unparameterized, no aggregation needed

            aggregate_type = 1;
            std::vector<int> _;
            std::vector<string> __;
            absRvar = AbstractRestrictedVariable(*it, _, __, it->name);

            if (test_duplicated.count(absRvar.ID) == 0) {
                Inputs_types.push_back(aggregate_type);
                dependent_abs_fluents.push_back(absRvar);
                test_duplicated.insert(absRvar.ID);
            } else {
                SystemUtils::abort("test_duplicated_0");
            }
        } else if (this->v2_contains_v1(it->patameters_types,
                                        mainAction.patameters_types)) {
            // partially share :aggregate_type =2
            if (it->patameters_types.size() == 1) {
                this->initialize_contains_helper(it, mainAction, 1,
                                                 test_duplicated, Inputs_types,
                                                 dependent_abs_fluents);
            } else {
                this->initialize_contains_helper(it, mainAction, 2,
                                                 test_duplicated, Inputs_types,
                                                 dependent_abs_fluents);
            }

        } else {
            this->initialize_contains_helper(it, mainAction, 2, test_duplicated,
                                             Inputs_types,
                                             dependent_abs_fluents);
        };
    };
}

void AttentionNet::set_action_modules_size(string aname, int size) {
    if (this->action_modules_sizes.count(aname) == 0) {
        (this->action_modules_sizes)[aname] = size;
    } else {
        int size2 = (this->action_modules_sizes)[aname];
        if (size != size2) {
            SystemUtils::abort("set action_modules_sizes mismatch. Got" +
                               to_string(size) + " and " + to_string(size2));
        }
    }
}

void AttentionNet::initialize_contains_helper(
    const set<AbstractVariable, classcomp>::iterator it,
    InstantiatedVariable& mainAction, int aggregate_type,
    set<string>& test_duplicated, vector<int>& Inputs_types,
    vector<AbstractRestrictedVariable>& dependent_abs_fluents) {
    vector<vector<int>> ObjectsRestriction_loc_s;
    vector<vector<string>> ObjectsRestriction_s;
    vector<string> ResVarIDs;

    for (int i = 0; i < it->patameters_types.size(); i++) {
        string tmp_type = (it->patameters_types)[i];
        vector<int> tmp1;
        vector<string> tmp2;
        vector<int> tmp3;
        vector<string> tmp4;
        for (int j = 0; j < mainAction.patameters_types.size(); j++) {
            if (mainAction.patameters_types[j] == tmp_type) {
                tmp1.push_back(i);
                tmp2.push_back(mainAction.instantiated_patameters[j]);
                tmp3.push_back(j);
                tmp4.push_back(mainAction.name);
            };
        };
        this->initialize_typeMatching_helper(
            i, aggregate_type, tmp1, tmp2, tmp3, tmp4, ResVarIDs,
            ObjectsRestriction_loc_s, ObjectsRestriction_s);
    };

    assert(ObjectsRestriction_loc_s.size() == ObjectsRestriction_s.size());
    assert(ObjectsRestriction_loc_s.size() == ResVarIDs.size());

    for (int i = 0; i < ObjectsRestriction_loc_s.size(); i++) {
        string id = it->name + ResVarIDs[i];
        bool skip = this->vec_is_intersect(mainAction.patameters_types,
                                           it->patameters_types) &&
                    (ResVarIDs[i] == "");
        if (!skip) {
            AbstractRestrictedVariable absRvar = AbstractRestrictedVariable(
                *it, ObjectsRestriction_loc_s[i], ObjectsRestriction_s[i], id);
            if (test_duplicated.count(absRvar.ID) == 0) {
                int aggregate_type2 = (it->patameters_types.size() >
                                       ObjectsRestriction_loc_s[i].size())
                                          ? 2
                                          : 1;
                // cout << it->name << "," << to_string(aggregate_type2) << ";"
                //      << to_string(ObjectsRestriction_loc_s.size()) << endl;
                // if (ObjectsRestriction_loc_s.size() > 0) {
                //     cout << ObjectsRestriction_loc_s[0] << endl;
                // };
                // cout << "===============" << endl;

                Inputs_types.push_back(aggregate_type2);
                dependent_abs_fluents.push_back(absRvar);
                test_duplicated.insert(absRvar.ID);
            } else {
                SystemUtils::abort("test_duplicated");
            }
        }
    };
}

void AttentionNet::initialize_readHelper(string filepath) {
    // set<string> ALL_patameters_types;
    std::ifstream taskFile;
    taskFile.open(filepath);
    if (taskFile.is_open()) {
        std::string line;
        set<string> duplicate_check;
        while (getline(taskFile, line)) {
            assert(line == "DIV");
            InstantiatedVariable var = InstantiatedVariable(taskFile);
            if (var.variable_type == "STATE_FLUENT") {
                this->StateFluents.push_back(var);
                AbstractVariable absvar = var;
                if (duplicate_check.count(absvar.name) == 0) {
                    this->StateFluentsSchema.insert(absvar);
                    duplicate_check.insert(absvar.name);
                    // for (int i = 0; i < absvar.patameters_types.size(); i++)
                    // {
                    //     ALL_patameters_types.insert(absvar.patameters_types[i]);
                    // }
                }

            } else if (var.variable_type == "ACTION_FLUENT") {
                this->ActionFluents.push_back(var);
                AbstractVariable absvar = var;
                if (duplicate_check.count(absvar.name) == 0) {
                    this->ActionFluentsSchema.insert(absvar);
                    duplicate_check.insert(absvar.name);
                    // for (int i = 0; i < absvar.patameters_types.size(); i++)
                    // {
                    //     ALL_patameters_types.insert(absvar.patameters_types[i]);
                    // }
                }

            } else if (var.variable_type == "NON_FLUENT") {
                this->NonFluents.push_back(var);
                AbstractVariable absvar = var;
                if (duplicate_check.count(absvar.name) == 0) {
                    this->NonFluentsSchema.insert(absvar);
                    duplicate_check.insert(absvar.name);
                    // for (int i = 0; i < absvar.patameters_types.size(); i++)
                    // {
                    //     ALL_patameters_types.insert(absvar.patameters_types[i]);
                    // }
                }
            } else {
                cout << "->" << var.variable_type << ". ";
                SystemUtils::abort("in initilase(), unkonwn type.");
            };
        };
        taskFile.close();
    };
    if (this->remove_file) {
        remove(filepath.c_str());
    };

    // Add no-op action
    // vector<string> all_params(ALL_patameters_types.begin(),
    //                           ALL_patameters_types.end());
    vector<string> _;
    vector<string> __;
    AbstractVariable absvar = AbstractVariable("no-op", "ACTION_FLUENT", _);
    this->ActionFluentsSchema.insert(absvar);
    InstantiatedVariable noop =
        InstantiatedVariable("no-op", "no-op", "ACTION_FLUENT", _, __);
    this->ActionFluents.push_back(noop);
}

void AttentionNet::initialize_Instantiation_helper(
    vector<InstantiatedVariable>& fluents_,
    AbstractRestrictedVariable& tmpARVar,
    vector<InstantiatedVariable>& Instantiated_ones) {
    for (int j = 0; j < fluents_.size(); j++) {
        if (tmpARVar.variable_type == fluents_[j].variable_type &&
            tmpARVar.name == fluents_[j].name) {
            bool fit = true;
            for (int jj = 0; jj < tmpARVar.ObjectsRestriction_loc.size();
                 jj++) {
                int loc = tmpARVar.ObjectsRestriction_loc[jj];
                string Robj = tmpARVar.ObjectsRestriction[jj];
                if (fluents_[j].patameters_types[loc] !=
                        tmpARVar.patameters_types[loc] ||
                    fluents_[j].instantiated_patameters[loc] != Robj) {
                    fit = false;
                };
            };
            if (fit) {
                Instantiated_ones.push_back(fluents_[j]);
            };
        }
    }
}

Fluent_descriptor_dependency AttentionNet::initialize_fluent_descriptor_helper(

    InstantiatedVariable& actionF, AbstractRestrictedVariable& Rfluent,
    InstantiatedVariable& Ifluent) {
    vector<AbstractRestrictedVariable> descriptor_dependent_abs_fluents;

    vector<int> Inputs_types;
    //
    vector<AbstractVariable> All_fluents;
    set<AbstractVariable, classcomp>::iterator it;
    for (it = this->StateFluentsSchema.begin();
         it != this->StateFluentsSchema.end(); it++) {
        All_fluents.push_back(*it);
    };
    for (it = this->NonFluentsSchema.begin();
         it != this->NonFluentsSchema.end(); it++) {
        All_fluents.push_back(*it);
    };
    //
    set<string> test_duplicated;
    for (size_t i = 0; i < All_fluents.size(); i++) {
        vector<vector<int>> ObjectsRestriction_loc_s;
        vector<vector<string>> ObjectsRestriction_s;
        vector<string> ResVarIDs;
        for (int j = 0; j < All_fluents[i].patameters_types.size(); j++) {
            string f_type = All_fluents[i].patameters_types[j];
            vector<int> tmp1;
            vector<string> tmp2;
            vector<int> tmp3;
            vector<string> tmp4;

            for (int k = 0; k < actionF.patameters_types.size(); k++) {
                if (f_type == actionF.patameters_types[k]) {
                    tmp1.push_back(j);
                    tmp2.push_back(actionF.instantiated_patameters[k]);
                    tmp3.push_back(k);
                    tmp4.push_back("a");
                }
            };
            for (int k = 0; k < Rfluent.patameters_types.size(); k++) {
                if (f_type == Rfluent.patameters_types[k]) {
                    tmp1.push_back(j);
                    tmp2.push_back(Ifluent.instantiated_patameters[k]);
                    tmp3.push_back(k);
                    tmp4.push_back("f");
                }
            };
            this->initialize_typeMatching_helper(
                j, 1, tmp1, tmp2, tmp3, tmp4, ResVarIDs,
                ObjectsRestriction_loc_s, ObjectsRestriction_s);
        };
        assert(ObjectsRestriction_loc_s.size() == ObjectsRestriction_s.size());
        assert(ObjectsRestriction_loc_s.size() == ResVarIDs.size());
        for (int j = 0; j < ObjectsRestriction_s.size(); j++) {
            if (!(ResVarIDs[j] == "")) {
                int aggregate_type = (All_fluents[i].patameters_types.size() >
                                      ObjectsRestriction_s[j].size())
                                         ? 2
                                         : 1;
                string id = All_fluents[i].name + ResVarIDs[j];
                AbstractRestrictedVariable absRvar = AbstractRestrictedVariable(
                    All_fluents[i], ObjectsRestriction_loc_s[j],
                    ObjectsRestriction_s[j], id);
                if (test_duplicated.count(absRvar.ID) == 0) {
                    Inputs_types.push_back(aggregate_type);
                    descriptor_dependent_abs_fluents.push_back(absRvar);
                    test_duplicated.insert(absRvar.ID);
                } else {
                    SystemUtils::abort("test_duplicated_2");
                }
            }
        }
    };
    // Instantiation
    // ID of AbstractRestrictedVariable -> instantiated ones
    map<string, vector<InstantiatedVariable>> descriptor_dependent_ins_fluents;
    this->initialize_Instantiation_(descriptor_dependent_abs_fluents,
                                    descriptor_dependent_ins_fluents);
    //
    this->set_fluent_descriptors_size(actionF.name, Rfluent.ID,
                                      descriptor_dependent_abs_fluents.size());

    Fluent_descriptor_dependency res = Fluent_descriptor_dependency(
        actionF, Ifluent, Rfluent, descriptor_dependent_abs_fluents,
        descriptor_dependent_ins_fluents, Inputs_types);
    return res;
}

void AttentionNet::set_fluent_descriptors_size(string action_name,
                                               string ResVarID, int size) {
    if ((this->fluent_descriptors_sizes).count(action_name) != 0) {
        if ((this->fluent_descriptors_sizes)[action_name].count(ResVarID) ==
            0) {
            (this->fluent_descriptors_sizes)[action_name][ResVarID] = size;
        } else {
            int size2 = (this->fluent_descriptors_sizes)[action_name][ResVarID];
            if (size != size2) {
                SystemUtils::abort("set_fluent_descriptors_size mismatch. Got" +
                                   to_string(size) + " and " +
                                   to_string(size2));
            }
        }
    } else {
        map<string, int> tmp;
        tmp[ResVarID] = size;
        (this->fluent_descriptors_sizes)[action_name] = tmp;
    }
}

void AttentionNet::initialize_typeMatching_helper(
    int const& param_index, int aggregate_type, vector<int>& tmp1,
    vector<string>& tmp2, vector<int>& tmp3, vector<string>& tmp4,
    vector<string>& ResVarIDs, vector<vector<int>>& ObjectsRestriction_loc_s,
    vector<vector<string>>& ObjectsRestriction_s) {
    if (param_index == 0) {
        for (int j = 0; j < tmp1.size(); j++) {
            vector<int> newVec1;
            vector<string> newVec2;
            newVec1.push_back(tmp1[j]);
            newVec2.push_back(tmp2[j]);
            string id = tmp4[j] + to_string(tmp3[j]);
            //  cout << "id:  " << id << endl;
            ObjectsRestriction_loc_s.push_back(newVec1);
            ObjectsRestriction_s.push_back(newVec2);
            ResVarIDs.push_back(id);
        }
        if (aggregate_type == 2) {
            vector<int> newVec1;
            vector<string> newVec2;
            string id = "";
            ObjectsRestriction_loc_s.push_back(newVec1);
            ObjectsRestriction_s.push_back(newVec2);
            ResVarIDs.push_back(id);
        }

    } else {
        vector<vector<int>> ObjectsRestriction_loc_sTMP;
        vector<vector<string>> ObjectsRestriction_sTMP;
        vector<string> ResVarIDsTMP;

        for (int ii = 0; ii < ObjectsRestriction_loc_s.size(); ii++) {
            for (int j = 0; j < tmp1.size(); j++) {
                vector<int> newVec1 = ObjectsRestriction_loc_s[ii];
                newVec1.push_back(tmp1[j]);

                vector<string> newVec2 = ObjectsRestriction_s[ii];
                newVec2.push_back(tmp2[j]);

                string id = ResVarIDs[ii];

                string newid = id + tmp4[j] + to_string(tmp3[j]);

                ObjectsRestriction_loc_sTMP.push_back(newVec1);
                ObjectsRestriction_sTMP.push_back(newVec2);
                ResVarIDsTMP.push_back(newid);
            };
            if (aggregate_type == 2) {
                vector<int> newVec1 = ObjectsRestriction_loc_s[ii];
                vector<string> newVec2 = ObjectsRestriction_s[ii];
                ObjectsRestriction_loc_sTMP.push_back(newVec1);
                ObjectsRestriction_sTMP.push_back(newVec2);
                string id = ResVarIDs[ii];
                ResVarIDsTMP.push_back(id);
            }
        };
        ObjectsRestriction_loc_s = ObjectsRestriction_loc_sTMP;
        ObjectsRestriction_s = ObjectsRestriction_sTMP;
        ResVarIDs = ResVarIDsTMP;
    }
}

void AttentionNet::initialize_Instantiation_(
    vector<AbstractRestrictedVariable>& dependent_abs_fluents,
    map<string, vector<InstantiatedVariable>>& dependent_ins_fluents) {
    for (int ii = 0; ii < dependent_abs_fluents.size(); ii++) {
        vector<InstantiatedVariable> Instantiated_ones;
        AbstractRestrictedVariable tmpARVar = dependent_abs_fluents[ii];
        this->initialize_Instantiation_helper(this->NonFluents, tmpARVar,
                                              Instantiated_ones);
        this->initialize_Instantiation_helper(this->StateFluents, tmpARVar,
                                              Instantiated_ones);
        dependent_ins_fluents[dependent_abs_fluents[ii].ID] = Instantiated_ones;
    };
}

void AttentionNet::initialize_modules_helper() {
    vector<InstantiatedVariable>::iterator it;
    set<string> Isdefined;

    for (it = (this->ActionFluents).begin(); it != (this->ActionFluents).end();
         it++) {
        if (Isdefined.count(it->name) == 0) {
            Isdefined.insert(it->name);
            int insize = (this->action_modules_sizes)[it->name];
            vector<int> topology;
            topology.push_back(insize);
            topology.push_back(insize - 1);
            topology.push_back(insize - 1);
            topology.push_back((int)(insize + 1) * 2 / 3);
            topology.push_back((int)(insize + 1) * 2 / 3);
            topology.push_back(1);

            if (this->ReadModulesParams) {
                auto subm = std::make_shared<nn_module>();
                subm = torch::nn::Module::register_module(
                    "action_module" + it->name, subm);
                (this->action_modules)[it->name] = *subm;
            } else {
                auto subm = std::make_shared<nn_module>(
                    "action_module" + it->name, topology);
                subm = torch::nn::Module::register_module(
                    "action_module" + it->name, subm);
                (this->action_modules)[it->name] = *subm;
            }

            vector<int> topology2;
            topology2.push_back(this->attention_layer_output_size);
            topology2.push_back(this->attention_layer_output_size - 1);
            topology2.push_back(1);
            if (this->ReadModulesParams) {
                auto subm2 = std::make_shared<nn_module>();
                subm2 = torch::nn::Module::register_module(
                    "action_modules_attentions" + it->name, subm2);
                (this->action_modules_attentions)[it->name] = *subm2;
            } else {
                auto subm2 = std::make_shared<nn_module>(
                    "action_modules_attentions" + it->name, topology2);
                subm2 = torch::nn::Module::register_module(
                    "action_modules_attentions" + it->name, subm2);
                (this->action_modules_attentions)[it->name] = *subm2;
            };

            map<string, map<string, Fluent_descriptor_dependency>> tmp =
                (this->fluent_descriptors_dependencies)[it->fullname];
            map<string, map<string, Fluent_descriptor_dependency>>::iterator
                it2;

            for (it2 = tmp.begin(); it2 != tmp.end(); it2++) {
                vector<int> topology3;
                int insize =
                    (this->fluent_descriptors_sizes)[it->name][it2->first];
                topology3.push_back(insize);
                int hiddensize =
                    (int)(this->attention_layer_output_size + insize) * 2 / 3;
                topology3.push_back(hiddensize);
                topology3.push_back(this->attention_layer_output_size);

                if (this->ReadModulesParams) {
                    auto subm3 = std::make_shared<nn_module>();
                    subm3 = torch::nn::Module::register_module(
                        "fluents_descriptors" + it->name + "," + it2->first,
                        subm3);
                    (this->fluents_descriptors)[it->name][it2->first] = *subm3;
                } else {
                    auto subm3 = std::make_shared<nn_module>(
                        "fluents_descriptors" + it->name + "," + it2->first,
                        topology3);
                    subm3 = torch::nn::Module::register_module(
                        "fluents_descriptors" + it->name + "," + it2->first,
                        subm3);
                    (this->fluents_descriptors)[it->name][it2->first] = *subm3;
                }

                vector<int> topology4;
                topology4.push_back(this->attention_layer_output_size);
                topology4.push_back(this->attention_layer_output_size - 1);
                topology4.push_back(1);
                if (this->ReadModulesParams) {
                    auto subm4 = std::make_shared<nn_module>();
                    subm4 = torch::nn::Module::register_module(
                        "aggregation_attentions" + it->name + "," + it2->first,
                        subm4);
                    (this->aggregation_attentions)[it->name][it2->first] =
                        *subm4;
                } else {
                    auto subm4 = std::make_shared<nn_module>(
                        "aggregation_attentions" + it->name + "," + it2->first,
                        topology4);
                    subm4 = torch::nn::Module::register_module(
                        "aggregation_attentions" + it->name + "," + it2->first,
                        subm4);
                    (this->aggregation_attentions)[it->name][it2->first] =
                        *subm4;
                };

                cout << "&&&" << it->name << "," << it2->first << endl;
            }
        }
    }
}

void AttentionNet::initialize_modules() {
    if (!this->ReadModulesParams) {
        this->initialize_modules_helper();
    } else {
        this->initialize_modules_helper();
        this->read_parameters();
    }
}

void AttentionNet::save_parameters(string path) {
    map<string, nn_module>::iterator it1;
    for (it1 = this->action_modules.begin(); it1 != this->action_modules.end();
         it1++) {
        string mname = "action_modules:[" + it1->first + "]";
        it1->second.save_submodel(path, mname);
    }

    map<string, map<string, nn_module>>::iterator it2;
    for (it2 = this->fluents_descriptors.begin();
         it2 != this->fluents_descriptors.end(); it2++) {
        for (it1 = it2->second.begin(); it1 != it2->second.end(); it2++) {
            string mname =
                "fluents_descriptors:[" + it2->first + "]+[" + it1->first + "]";
            it1->second.save_submodel(path, mname);
        }
    }

    for (it1 = this->action_modules_attentions.begin();
         it1 != this->action_modules_attentions.end(); it1++) {
        string mname = "action_modules_attentions:[" + it1->first + "]";
        it1->second.save_submodel(path, mname);
    }

    for (it2 = this->aggregation_attentions.begin();
         it2 != this->aggregation_attentions.end(); it2++) {
        for (it1 = it2->second.begin(); it1 != it2->second.end(); it2++) {
            string mname = "aggregation_attentions:[" + it2->first + "]+[" +
                           it1->first + "]";
            it1->second.save_submodel(path, mname);
        }
    }
}

void AttentionNet::read_parameters() {
    map<string, nn_module>::iterator it1;
    for (it1 = this->action_modules.begin(); it1 != this->action_modules.end();
         it1++) {
        string mname = "action_modules:[" + it1->first + "]";
        it1->second.Load_submodel(this->ReadModulesParams_Path, mname);
    }

    map<string, map<string, nn_module>>::iterator it2;
    for (it2 = this->fluents_descriptors.begin();
         it2 != this->fluents_descriptors.end(); it2++) {
        for (it1 = it2->second.begin(); it1 != it2->second.end(); it2++) {
            string mname =
                "fluents_descriptors:[" + it2->first + "]+[" + it1->first + "]";
            it1->second.Load_submodel(this->ReadModulesParams_Path, mname);
        }
    }

    for (it1 = this->action_modules_attentions.begin();
         it1 != this->action_modules_attentions.end(); it1++) {
        string mname = "action_modules_attentions:[" + it1->first + "]";
        it1->second.Load_submodel(this->ReadModulesParams_Path, mname);
    }

    for (it2 = this->aggregation_attentions.begin();
         it2 != this->aggregation_attentions.end(); it2++) {
        for (it1 = it2->second.begin(); it1 != it2->second.end(); it2++) {
            string mname = "aggregation_attentions:[" + it2->first + "]+[" +
                           it1->first + "]";
            it1->second.Load_submodel(this->ReadModulesParams_Path, mname);
        }
    }
}

void AttentionNet::change_instance(AttentionNet& a2) {
    assert(a2.attention_layer_output_size = this->attention_layer_output_size);
    this->StateFluents = a2.StateFluents;
    this->NonFluents = a2.NonFluents;
    this->ActionFluents = a2.ActionFluents;
    this->action_modules_dependencies = a2.action_modules_dependencies;
    this->fluent_descriptors_dependencies = a2.fluent_descriptors_dependencies;
    this->stateVariableIndices_net = a2.stateVariableIndices_net;
    this->action_indices_ = a2.action_indices_;
    this->res_order = a2.res_order;
}

void AttentionNet::save_indices() {
    string fileName = "Indices";
    std::ofstream taskFile;
    taskFile.open(fileName.c_str());

    map<string, int>::iterator it;
    taskFile << "stateVariableIndices_net" << endl;
    for (it = this->stateVariableIndices_net.begin();
         it != this->stateVariableIndices_net.end(); it++) {
        taskFile << it->first << endl;
        taskFile << it->second << endl;
    }
    taskFile << "end" << endl;

    taskFile << "action_indices_" << endl;
    for (it = this->action_indices_.begin(); it != this->action_indices_.end();
         it++) {
        taskFile << it->first << endl;
        taskFile << it->second << endl;
    }
    taskFile << "end" << endl;
    taskFile.close();
}

void AttentionNet::read_indices(string path) {
    string fileName = "Indices";
    std::ifstream taskFile;
    string taskFileName = path + fileName;
    taskFile.open(taskFileName.c_str());
    if (taskFile.is_open()) {
        std::string line;
        getline(taskFile, line);
        assert(line == "stateVariableIndices_net");
        getline(taskFile, line);
        while (line != "end") {
            string key = line;
            getline(taskFile, line);
            int v = stoi(line);
            this->stateVariableIndices_net[key] = v;
            getline(taskFile, line);
        }

        getline(taskFile, line);
        assert(line == "action_indices_");
        getline(taskFile, line);
        while (line != "end") {
            string key = line;
            getline(taskFile, line);
            int v = stoi(line);
            this->action_indices_[key] = v;
            getline(taskFile, line);
        }

        taskFile.close();
    }
}

bool AttentionNet::check_domain_schema(
    set<AbstractVariable, classcomp> const& StateFluentsSchema1,
    set<AbstractVariable, classcomp> const& NonFluentsSchema1,
    set<AbstractVariable, classcomp> const& ActionFluentsSchema1,
    map<string, int> const& action_modules_sizes1,
    map<string, map<string, int>> const& fluent_descriptors_sizes1,
    set<AbstractVariable, classcomp> const& StateFluentsSchema2,
    set<AbstractVariable, classcomp> const& NonFluentsSchema2,
    set<AbstractVariable, classcomp> const& ActionFluentsSchema2,
    map<string, int> const& action_modules_sizes2,
    map<string, map<string, int>> const& fluent_descriptors_sizes2) {
    assert(StateFluentsSchema1.size() == StateFluentsSchema2.size());
    assert(NonFluentsSchema1.size() == NonFluentsSchema2.size());
    assert(ActionFluentsSchema1.size() == ActionFluentsSchema2.size());

    set<AbstractVariable, classcomp>::iterator it;
    set<AbstractVariable, classcomp>::iterator it2;
    it = StateFluentsSchema1.begin();
    it2 = StateFluentsSchema2.begin();
    while (it != StateFluentsSchema1.end()) {
        assert(*it == *it2);
        it++;
        it2++;
    };
    it = NonFluentsSchema1.begin();
    it2 = NonFluentsSchema2.begin();
    while (it != NonFluentsSchema1.end()) {
        assert(*it == *it2);
        it++;
        it2++;
    }
    it = ActionFluentsSchema1.begin();
    it2 = ActionFluentsSchema2.begin();
    while (it != ActionFluentsSchema1.end()) {
        assert(*it == *it2);
        it++;
        it2++;
    }
    map<string, int>::iterator it3;
    for (it3 = action_modules_sizes1.begin();
         it3 != action_modules_sizes1.end(); it3++) {
        assert(it3->second == action_modules_sizes2[it3->first]);
    }

    map<string, map<string, int>>::iterator it4;
    for (it4 = fluent_descriptors_sizes1.begin();
         it4 != fluent_descriptors_sizes1.end(); it4++) {
        map<string, int> tmp = fluent_descriptors_sizes2[it4->first];
        assert(tmp.size() == it4->second.size());
        for (it3 = it4->second.begin(); it3 != it4->second.end(); it3++) {
            assert(it3->second == tmp[it3->first]);
        }
    }

    return true;
}

bool AttentionNet::vec_is_intersect(vector<string> v1, vector<string> v2) {
    for (size_t i = 0; i < v1.size(); i++) {
        for (size_t i2 = 0; i2 < v2.size(); i2++) {
            if (v1[i] == v2[i2]) {
                return true;
            }
        }
    }
    return false;
}

bool AttentionNet::v2_contains_v1(vector<string> v1, vector<string> v2) {
    for (size_t i = 0; i < v1.size(); i++) {
        bool res2 = false;
        for (size_t j = 0; j < v2.size(); j++) {
            if (v1[i] == v2[j]) {
                res2 = true;
            };
        };
        if (!res2) {
            return false;
        };
    }
    return true;
}

bool AttentionNet::v1_contains_diff_from_v2(vector<string> v1,
                                            vector<string> v2) {
    for (size_t i = 0; i < v1.size(); i++) {
        bool r = false;
        for (size_t j = 0; j < v2.size(); j++) {
            if (v1[i] == v2[j]) {
                r = true;
            }
        }
        if (!r) {
            return true;
        }
    }
    return false;
}
