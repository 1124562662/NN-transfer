#include "neuralMLN.h"
#include "Variables.h"
#include "assert.h"
#include "dependency.h"
#include "utils/math_utils.h"
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
#include <torch/torch.h>
using namespace std;

torch::Tensor neuralMLN::forward(vector<double> const& current_state_) {
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
    torch::Tensor res;
    return res;
}

void neuralMLN::initialize(string file) {
    this->initialize_readHelper1(file);
    vector<AbstractVariable> vecVars;
    set<AbstractVariable, classcomp>::iterator it;
    it = this->StateFluentsSchema.begin();
    while (it != this->StateFluentsSchema.end()) {
        vecVars.push_back(*it);
        it++;
    };
    it = this->NonFluentsSchema.begin();
    while (it != this->NonFluentsSchema.end()) {
        vecVars.push_back(*it);
        it++;
    };

    this->mln_ = myMLN(vecVars);

    for (int i = 0; i < this->filter_size; i += 2) {
        (this->Fliters)[i] = this->mln_.perform_random_walk(i + 1, "DFS");
        (this->Fliters)[i + 1] = this->mln_.perform_random_walk(i + 1, "DFS");
    };

    map<int, filter>::iterator it2 = this->Fliters.begin();

    while (it2 != this->Fliters.end()) {
        vector<InstantiatedVariable> _;

        for (size_t i = 0; i < it2->second.absVars.size(); i++) {
            cout << it2->second.absVars[i].name << " (";
            cout << it2->second.restrictions[i][0].first.first << ","
                 << it2->second.restrictions[i][0].first.second << ")(";
            cout << it2->second.restrictions[i][0].second.first << ","
                 << it2->second.restrictions[i][0].second.second << ")";
            for (size_t i1 = 0;
                 i1 < it2->second.absVars[i].patameters_types.size(); i1++) {
                cout << " " << it2->second.absVars[i].patameters_types[i1];
            }

            cout << ", ";
        };
        cout << "....." << endl;
        cout << "" << endl;

        vector<vector<InstantiatedVariable>> res;
        set<string> duplictes;
        this->ins_helper(duplictes, it2->second.absVars, _,
                         it2->second.restrictions, res);

        if (res.size() == 0) {
            cout << "empty initializatoin for " + to_string(it2->first) << endl;

        } else {
            for (size_t i = 0; i < res[0].size(); i++) {
                cout << res[0][i].fullname << ", ";
            };
            cout << ".." << endl;
            cout << "" << endl;
        }

        cout << to_string(res.size()) << endl;
        cout << "" << endl;

        cout << "" << endl;
        this->InstantiatedInputs[it2->first] = res;
        it2++;
    }

    it2 = this->Fliters.begin();
    while (it2 != this->Fliters.end()) {
       
           int insize = it2->second.absVars.size();
            vector<int> topology;
            topology.push_back(insize);
            topology.push_back(this->attention_layer_output_size);

            // if (this->ReadModulesParams) {
            //     auto subm = std::make_shared<nn_module>();
            //     subm = torch::nn::Module::register_module(
            //         "f" + it->name, subm);
            //     (this->action_modules)[it->name] = *subm;
            // } else {
            //     auto subm = std::make_shared<nn_module>(
            //         "f" + it->name, topology);
            //     subm = torch::nn::Module::register_module(
            //         "f" + it->name, subm);
            //     (this->action_modules)[it->name] = *subm;
            // }
          
            
           
        
        it2++;
    }
    if(this->ReadModulesParams){
       //  this->read_parameters();
    }
}

void neuralMLN::ins_helper(
    set<string>& duplictes, vector<AbstractVariable> remaining,
    vector<InstantiatedVariable> Instantiated_ones,
    vector<vector<pair<pair<int, int>, pair<int, int>>>>& restriction_s,
    vector<vector<InstantiatedVariable>>& result) {
    if (remaining.size() == 0) {
        string c = "";
        for (size_t i = 0; i < Instantiated_ones.size(); i++) {
            c += Instantiated_ones[i].fullname;
        }
        if (duplictes.count(c) == 0) {
            result.push_back(Instantiated_ones);
        }

        return;
    }

    AbstractVariable a = remaining[0];
    vector<AbstractVariable> remaining2 =
        vector<AbstractVariable>(remaining.begin() + 1, remaining.end());

    vector<InstantiatedVariable> inss =
        find_ins(a, restriction_s, Instantiated_ones, Instantiated_ones.size());

    for (int i = 0; i < inss.size(); i++) {
        vector<InstantiatedVariable> tmp = Instantiated_ones;
        tmp.push_back(inss[i]);
        this->ins_helper(duplictes, remaining2, tmp, restriction_s, result);
    }
    return;
}

vector<InstantiatedVariable> neuralMLN::find_ins(
    AbstractVariable a,
    vector<vector<pair<pair<int, int>, pair<int, int>>>>& restriction_s,
    vector<InstantiatedVariable>& Instantiated_ones, int idx) {
    vector<string> vp;
    vector<int> v2;

    if (idx > 0) {
        for (int i = 0; i < restriction_s[idx].size(); i++) {
            pair<int, int> p11 = restriction_s[idx][i].first;
            pair<int, int> p22 = restriction_s[idx][i].second;
            if (Instantiated_ones.size() <= p11.first) {
                cout << to_string(Instantiated_ones.size())
                     << to_string(p11.first) << endl;
                SystemUtils::abort("a1");
            }
            if (Instantiated_ones[p11.first].instantiated_patameters.size() <=
                p11.second) {
                SystemUtils::abort("a2");
            }

            string resSTR = Instantiated_ones[p11.first]
                                .instantiated_patameters[p11.second];

            vp.push_back(resSTR);
            v2.push_back(p22.second);
        }

        assert(v2.size() > 0);
    }

    vector<InstantiatedVariable> res;
    vector<InstantiatedVariable>::iterator it;
    for (it = StateFluents.begin(); it != StateFluents.end(); it++) {
        if ((AbstractVariable)(*it) == a) {
            // cout << (*it).fullname << "," << a.name << endl;

            if (idx == 0) {
                res.push_back(*it);
            } else {
                bool fit = true;
                for (size_t i = 0; i < v2.size(); i++) {
                    if (it->instantiated_patameters[v2[i]] != vp[i]) {
                        fit = false;
                    }
                }
                if (fit) {
                    //   cout<<it->instantiated_patameters[v2[0]]<<","<<vp[0]<<endl;
                    res.push_back(*it);
                };
            };
        };
    };

    for (it = NonFluents.begin(); it != NonFluents.end(); it++) {
        if ((AbstractVariable)(*it) == a) {
            // cout << (*it).fullname << "," << a.name << endl;

            if (Instantiated_ones.size() == 0) {
                res.push_back(*it);
            } else {
                bool fit = true;
                for (size_t i = 0; i < v2.size(); i++) {
                    if (it->instantiated_patameters[v2[i]] != vp[i]) {
                        fit = false;
                    }
                }
                if (fit) {
                    // cout<<it->instantiated_patameters[v2[0]]<<","<<vp[0]<<endl;
                    res.push_back(*it);
                };
            };
        };
    };
    return res;
}

void neuralMLN::initialize_readHelper1(string filepath) {
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
        if (remove(filepath.c_str()) != 0) {
            SystemUtils::abort("remove");
        };
    };
}

bool neuralMLN::check_domain_schema(
    set<AbstractVariable, classcomp> const& StateFluentsSchema1,
    set<AbstractVariable, classcomp> const& NonFluentsSchema1,
    set<AbstractVariable, classcomp> const& ActionFluentsSchema1,
    set<AbstractVariable, classcomp> const& StateFluentsSchema2,
    set<AbstractVariable, classcomp> const& NonFluentsSchema2,
    set<AbstractVariable, classcomp> const& ActionFluentsSchema2) {
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

    return true;
}

bool neuralMLN::vec_is_intersect(vector<string> v1, vector<string> v2) {
    for (size_t i = 0; i < v1.size(); i++) {
        for (size_t i2 = 0; i2 < v2.size(); i2++) {
            if (v1[i] == v2[i2]) {
                return true;
            }
        }
    }
    return false;
}

bool neuralMLN::v2_contains_v1(vector<string> v1, vector<string> v2) {
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

bool neuralMLN::v1_contains_diff_from_v2(vector<string> v1, vector<string> v2) {
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
