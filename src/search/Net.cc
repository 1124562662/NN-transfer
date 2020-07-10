#include "Net.h"
#include <torch/torch.h>

#include "utils/math_utils.h"

#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <algorithm>
#include <iostream>

#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>

using namespace std;

// need to be called
// can be called after planer's ini_session()
void Net::save_parameters(std::string filepath) {
    std::ofstream taskFile;
    taskFile.open(filepath + "saved_parameters");

    //(1) int layer
    taskFile << "layer" << std::endl;
    taskFile << this->layer << std::endl;
    taskFile << "end" << std::endl;

    //(2) action_indices_
    // std::map<std::string, int>
    taskFile << "action_indices_" << std::endl;
    std::map<std::string, int>::iterator it1;
    for (it1 = this->action_indices_->begin();
         it1 != this->action_indices_->end(); it1++) {
        taskFile << it1->first << std::endl;
        taskFile << it1->second << std::endl;
    }
    taskFile << "end" << std::endl;

    //(3) action_f_net
    // std::vector<std::string>
    taskFile << "action_f_net" << std::endl;
    for (int y = 0; y < this->action_f_net.size(); y++) {
        taskFile << (this->action_f_net)[y] << std::endl;
    }
    taskFile << "end" << std::endl;

    //(4)  stateVariableIndices_net
    // std::map<std::string, int>
    taskFile << "stateVariableIndices_net" << std::endl;
    for (it1 = this->stateVariableIndices_net->begin();
         it1 != this->stateVariableIndices_net->end(); it1++) {
        taskFile << it1->first << std::endl;
        taskFile << it1->second << std::endl;
    }
    taskFile << "end" << std::endl;

    //(5) sub_models_dependency_information_
    // std::map<std::string, dependency_info>
    taskFile << "sub_models_dependency_information_" << std::endl;
    std::map<std::string, dependency_info>::iterator it2;
    for (it2 = (this->sub_models_dependency_information_).begin();
         it2 != (this->sub_models_dependency_information_).end(); it2++) {
        taskFile << "statefluent:" << std::endl;
        taskFile << it2->first << std::endl;

        // 5.1  name_  string
        taskFile << "name_" << std::endl;
        taskFile << it2->second.name_ << std::endl;

        // 5.2   dependent_SFs
        //   std::vector<std::string>
        taskFile << "dependent_SFs" << std::endl;
        for (int y = 0; y < it2->second.dependent_SFs.size(); y++) {
            taskFile << it2->second.dependent_SFs[y] << std::endl;
        }
        taskFile << "end" << std::endl;

        // 5.3 instantiated_quantifier_SFs_subM
        // std::map<std::string, std::vector<std::string>>
        taskFile << "instantiated_quantifier_SFs_subM" << std::endl;
        std::map<std::string, std::vector<std::string>>::iterator it3;
        for (it3 = (it2->second).instantiated_quantifier_SFs_subM.begin();
             it3 != (it2->second).instantiated_quantifier_SFs_subM.end();
             it3++) {
            taskFile << "first" << std::endl;
            taskFile << it3->first << std::endl;
            taskFile << "second" << std::endl;
            for (int p = 0; p < it3->second.size(); p++) {
                taskFile << (it3->second)[p] << std::endl;
            }
            taskFile << "end" << std::endl;
        }
        taskFile << "end" << std::endl;

        // 5.4 order_of_quantifier_subM
        // std::map<std::string, std::vector<std::string>>
        taskFile << "order_of_quantifier_subM" << std::endl;
        for (it3 = (it2->second).order_of_quantifier_subM.begin();
             it3 != (it2->second).order_of_quantifier_subM.end(); it3++) {
            taskFile << "first" << std::endl;
            taskFile << it3->first << std::endl;
            taskFile << "second" << std::endl;
            for (int p = 0; p < it3->second.size(); p++) {
                taskFile << (it3->second)[p] << std::endl;
            }
            taskFile << "end" << std::endl;
        }
        taskFile << "end" << std::endl;

        // 5.5  remaining_SFs
        // std::map<std::string, std::set<std::string>>
        std::map<std::string, std::set<std::string>>::iterator it4;

        taskFile << "remaining_SFs" << std::endl;
        for (it4 = (it2->second).remaining_SFs.begin();
             it4 != (it2->second).remaining_SFs.end(); it4++) {
            taskFile << "first" << std::endl;
            taskFile << it4->first << std::endl;
            taskFile << "second" << std::endl;
            std::set<std::string>::iterator it5 = it4->second.begin();
            while (it5 != it4->second.end()) {
                taskFile << *it5 << std::endl;
                it5++;
            }
            taskFile << "end" << std::endl;
        }
        taskFile << "end" << std::endl;
    }
    taskFile << "end" << std::endl;

    // (6) torch_sub_models
    // std::map<int ,std::map<std::string , sub_model>>
    taskFile << "torch_sub_models" << std::endl;
    std::map<int, std::map<std::string, sub_model>>::iterator it4;
    std::map<std::string, sub_model>::iterator it5;
    for (it4 = this->torch_sub_models.begin();
         it4 != this->torch_sub_models.end(); it4++) {
        taskFile << "layer_int" << std::endl;
        taskFile << it4->first << std::endl;

        for (it5 = (it4->second).begin(); it5 != (it4->second).end(); it5++) {
            std::string model_path = filepath + "savedModels/" + it5->first +
                                     "+" + std::to_string(it4->first) + "+";
            sub_model* tmp_m =
                &(this->torch_sub_models[it4->first][it5->first]);
            tmp_m->save_submodel(model_path);
            // abstract sf name  it5->first
            // layer  it4->first
            taskFile << "abs_name" << std::endl;
            taskFile << it5->first << std::endl;
        }

        taskFile << "end" << std::endl;
    }

    taskFile << "end" << std::endl;
    taskFile << "end" << std::endl;
    taskFile.close();

    //     struct sub_model Seq =sub_model(8);
    //      std::cout << Seq.forward(torch::zeros({1, 7})) << std::endl;
    //     torch::serialize::OutputArchive output_archive;
    //     Seq.save(output_archive);
    //     output_archive.save_to("savedModels1/person-waiting-down2qq.pt");

    //  torch::serialize::InputArchive input_archive1;
    //     struct sub_model savedSeq;
    //     input_archive1.load_from("savedModels1/person-waiting-down2qq.pt");
    //     savedSeq.load(input_archive1);
}

void Net::initialize_(bool read_parameters, std::string path,
                      int output_size_subM) {
    //---------------------set size sub --------------------------------//
    int submodule_size_sf = 5;
    int submodule_size_af = 7;
    //---------------------set size sub --------------------------------//
    std::map<std::string, double>::iterator niter;
    for (niter = this->NonfluentsToValues_net->begin();
         niter != this->NonfluentsToValues_net->end(); niter++) {
        this->Nonfluents_types.insert(this->get_variable_name(niter->first));
    }

    if (!read_parameters) {
        this->output_size_subM = output_size_subM;
        // initilize the sf sub models
        this->initialize_sf_subMod(submodule_size_sf);
        std::cout << "state fluents' modules finishied initialization."
                  << std::endl;
        // initialize the final action layer;
        this->initialize_action_subMod(submodule_size_af);
        std::cout << "action fluents' modules finishied initialization."
                  << std::endl;

    } else {
        cout << "reading parameters start...." << endl;
        cout << "reading parameters start...." << endl;
        cout << "reading parameters start...." << endl;
        cout << "reading parameters start...." << endl;
        cout << "reading parameters start...." << endl;
        cout << "reading parameters start...." << endl;
        cout << "reading parameters start...." << endl;
        this->output_size_subM = output_size_subM;
        //(1) read in saved_parameters
        std::ifstream taskFile;
        taskFile.open(path + "saved_parameters");
        if (taskFile.is_open()) {
            std::string line;
            while (getline(taskFile, line)) {
                std::cout << "line" << line << std::endl;

                if (line == "layer") {
                    getline(taskFile, line);

                    this->layer = std::stoi(line);
                    getline(taskFile, line);

                } else if (line == "action_indices_") {
                    getline(taskFile, line);
                    while (line != "end") {
                        std::string key = line;
                        getline(taskFile, line);
                        (*(this->action_indices_))[key] = std::stoi(line);
                        getline(taskFile, line);
                    }

                } else if (line == "action_f_net") {
                    getline(taskFile, line);
                    while (line != "end") {
                        (this->action_f_net).push_back(line);
                        getline(taskFile, line);
                    }
                } else if (line == "stateVariableIndices_net") {
                    getline(taskFile, line);
                    while (line != "end") {
                        std::string key = line;
                        getline(taskFile, line);
                        (*(this->stateVariableIndices_net))[key] =
                            std::stoi(line);
                        getline(taskFile, line);
                    }

                } else if (line == "sub_models_dependency_information_") {
                    getline(taskFile, line); // statefluent:

                    while (line != "end") {
                        assert(line == "statefluent:");
                        getline(taskFile, line);
                        std::string mainkey = line;

                        getline(taskFile, line); // "name_"
                        assert(line == "name_");
                        getline(taskFile, line);
                        std::string name_ = line;

                        std::vector<std::string> dependent_SFs;
                        getline(taskFile, line); //"dependent_SFs"
                        assert(line == "dependent_SFs");
                        getline(taskFile, line);
                        while (line != "end") {
                            dependent_SFs.push_back(line);
                            getline(taskFile, line);
                        }

                        //// 5.3 instantiated_quantifier_SFs_subM
                        getline(taskFile, line);
                        assert(line == "instantiated_quantifier_SFs_subM");
                        std::map<std::string, std::vector<std::string>>
                            instantiated_quantifier_SFs_subM;
                        getline(taskFile, line);
                        while (line != "end") {
                            assert(line == "first");
                            getline(taskFile, line);
                            std::string key = line;
                            getline(taskFile, line);
                            assert(line == "second");
                            std::vector<std::string> vec;
                            getline(taskFile, line);
                            while (line != "end") {
                                vec.push_back(line);
                                getline(taskFile, line);
                            }

                            instantiated_quantifier_SFs_subM[key] = vec;
                            getline(taskFile, line);
                        }

                        // 5.4 order_of_quantifier_subM
                        getline(taskFile, line);
                        assert(line == "order_of_quantifier_subM");
                        std::map<std::string, std::vector<std::string>>
                            order_of_quantifier_subM;
                        getline(taskFile, line);
                        while (line != "end") {
                            assert(line == "first");
                            getline(taskFile, line);
                            std::string key = line;

                            getline(taskFile, line);
                            assert(line == "second");
                            std::vector<std::string> vec;
                            getline(taskFile, line);
                            while (line != "end") {
                                vec.push_back(line);
                                getline(taskFile, line);
                            }
                            order_of_quantifier_subM[key] = vec;
                            getline(taskFile, line);
                        }

                        // 5.5 "remaining_SFs"
                        getline(taskFile, line);
                        assert(line == "remaining_SFs");

                        std::map<std::string, std::set<std::string>>
                            remaining_SFs;
                        getline(taskFile, line);
                        while (line != "end") {
                            assert(line == "first");
                            getline(taskFile, line);
                            std::string key = line;

                            getline(taskFile, line);
                            assert(line == "second");
                            std::set<std::string> se;
                            getline(taskFile, line);
                            while (line != "end") {
                                se.insert(line);
                                getline(taskFile, line);
                            }
                            remaining_SFs[key] = se;
                            getline(taskFile, line);
                        }

                        getline(taskFile, line); // statefluent or end

                        dependency_info depend_info = dependency_info(
                            name_, dependent_SFs,
                            instantiated_quantifier_SFs_subM,
                            order_of_quantifier_subM, remaining_SFs);
                        (this->sub_models_dependency_information_)[mainkey] =
                            depend_info;
                    };
                } else if (line == "torch_sub_models") {
                    getline(taskFile, line);
                    while (line != "end") {
                        assert(line == "layer_int");
                        getline(taskFile, line);
                        int keyint = std::stoi(line);
                        getline(taskFile, line);
                        while (line != "end") {
                            assert(line == "abs_name");
                            getline(taskFile, line);
                            std::string absname = line;
                            // initialize the submodel
                            std::string model_path =
                                path + "savedModels/" + absname + "+" +
                                std::to_string(keyint) + "+";
                            int tmp_size = (this->check_if_is_AF(absname))
                                               ? submodule_size_af
                                               : submodule_size_sf;
                            if (absname == "no-op") {
                                tmp_size = submodule_size_af;
                            }
                            auto subm = std::make_shared<sub_model>(tmp_size);
                            subm->initialize(model_path, tmp_size - 1);
                            subm = torch::nn::Module::register_module(
                                absname + std::to_string(keyint), subm);
                            (this->torch_sub_models)[keyint][absname] = *subm;
                            //  (this->torch_sub_models)[keyint].insert (
                            //  std::pair<std::string,sub_model>(absname,*subm)
                            //  );

                            //
                            getline(taskFile, line);
                        }

                        getline(taskFile, line);
                    }
                } else {
                    assert(line == "end");
                    if (line != "end") {
                        std::cout << "error" << line << std::endl;
                    }
                }
            }
        }
        taskFile.close();
    }

    std::vector<std::string> res_order_(this->action_indices_->size() + 1);
    std::map<std::string, int>::iterator itt;
    for (itt = this->action_indices_->begin();
         itt != this->action_indices_->end(); itt++) {
        res_order_[itt->second] = itt->first;
    }
    this->res_order = res_order_;

    value_set = true;
}

void Net::initialize_(
    bool read_parameters, std::string filename_, int output_size_subM,
    std::map<std::string, double>* NonfluentsToValues_net,
    std::map<std::string, std::map<std::string, std::set<std::string>>>*
        sf2VariedSize_sfs_net,
    std::map<std::string, std::set<std::string>>* sf2FixedSize_sfs_net,
    std::map<std::string, int>* stateVariableIndices_net,
    std::vector<std::vector<std::string>>* stateVariableValues_net,
    std::vector<std::string> action_f_net,
    std::map<std::string, int>* action_indices_

) {
    this->NonfluentsToValues_net = NonfluentsToValues_net;
    this->sf2VariedSize_sfs_net = sf2VariedSize_sfs_net;
    this->sf2FixedSize_sfs_net = sf2FixedSize_sfs_net;
    this->stateVariableIndices_net = stateVariableIndices_net;

    this->stateVariableValues_net = stateVariableValues_net;
    this->action_f_net = action_f_net;
    this->action_indices_ = action_indices_;
    this->initialize_(read_parameters, filename_, output_size_subM);
}
void Net::initialize_sf_subMod(int submodule_size) {
    // abstract sf name to the abstract sf with parameters
    std::map<std::string, std::string> sf2sf_wirhP;
    // construct sf2sf_wirhP
    std::map<std::string, std::set<std::string>>::iterator it1;
    for (it1 = this->sf2FixedSize_sfs_net->begin();
         it1 != this->sf2FixedSize_sfs_net->end(); it1++) {
        sf2sf_wirhP[this->get_variable_name(it1->first)] = it1->first;
    }
    // contruct non_fluents_types
    std::set<std::string> non_fluents_types;
    std::map<std::string, double>::iterator itkt;
    for (itkt = this->NonfluentsToValues_net->begin();
         itkt != this->NonfluentsToValues_net->end(); itkt++) {
        non_fluents_types.insert(this->get_variable_name(itkt->first));
    }

    // construct the state-fluent module and their dependency information
    std::map<std::string, int>::iterator it;
    for (it = this->stateVariableIndices_net->begin();
         it != this->stateVariableIndices_net->end(); it++) {
        int input_size_subm =
            0; // used to record the number of type of instantiated sf
        int input_size_subm_zero = 0;
        std::set<std::string> used;

        std::string sf_ = it->first;
        std::string sf_name = this->get_variable_name(sf_);
        std::vector<std::string> params_instance = this->get_param_vec(sf_);
        ;
        std::string sf_abstract = sf2sf_wirhP[sf_name];
        std::vector<std::string> params_abstract =
            this->get_param_vec(sf_abstract);

        // (1) instantiate the parameters of (fixed size) dependent
        std::set<std::string>* dependent_SFs =
            &(this->sf2FixedSize_sfs_net->operator[](sf_abstract));

        // the things that wanted
        std::vector<std::string> dependent_SFs_instantiated; // res
        std::set<std::string>::iterator it3;

        for (it3 = dependent_SFs->begin(); it3 != dependent_SFs->end(); it3++) {
            std::vector<std::string> dependent_params_abstract =
                this->get_param_vec(*it3);
            std::vector<std::string> dependent_params_instantiated =
                dependent_params_abstract;

            // instantiate the parameters
            for (unsigned int j = 0; j < dependent_params_abstract.size();
                 ++j) {
                for (unsigned int jj = 0; jj < params_abstract.size(); ++jj) {
                    if (params_abstract[jj] == dependent_params_abstract[j]) {
                        dependent_params_instantiated[j] = params_instance[jj];
                    }
                }
            }
            // need to remove all the " " char
            std::string instantietaed_dependent_SF =
                this->get_variable_name(*it3);
            instantietaed_dependent_SF += "(";
            for (unsigned int j = 0; j < dependent_params_instantiated.size();
                 ++j) {
                instantietaed_dependent_SF += dependent_params_instantiated[j];
                if (j != dependent_params_instantiated.size() - 1) {
                    instantietaed_dependent_SF += ",";
                }
            }
            instantietaed_dependent_SF += ")";
            instantietaed_dependent_SF =
                this->erasechar(instantietaed_dependent_SF);
            if (!(this->check_if_is_AF(instantietaed_dependent_SF))) {
                dependent_SFs_instantiated.push_back(
                    instantietaed_dependent_SF);
                used.insert(this->get_variable_name(*it3));
                input_size_subm_zero += 1;
                this->add_input_size_subm(*it3, input_size_subm);
            }
        }

        // (2) instantiate SFs that involved in quantifiers

        // quantifier -> vector of abstract dependent fluents
        std::map<std::string, std::vector<std::string>> order_of_quantifier_;

        // abstract_Sf -> vector of instantiated sf  (same type but with
        // different instantiated
        std::map<std::string, std::vector<std::string>>
            instantiated_quantifier_SFs_;

        std::map<std::string, std::set<std::string>>::iterator it4;
        // it4->first is quantifier, it4->second is set
        for (it4 = (*(this->sf2VariedSize_sfs_net))[sf_abstract].begin();
             it4 != (*(this->sf2VariedSize_sfs_net))[sf_abstract].end();
             it4++) {
            std::set<std::string>::iterator it5 = it4->second.begin();
            // vector of abstract dependent fluents
            std::vector<std::string> dependent_quantifir_sf_vec;
            while (it5 != it4->second.end()) { // *it5 is abstract SF
                std::string param_quantifier;
                std::string param_quantifier_type;
                if (((*it5).find(':') != std::string::npos) ||
                    (this->check_if_is_AF(
                        this->erasechar(*it5)))) { // is an action fluent
                } else {
                    // vector of instantiated Sfs
                    std::vector<std::string> intantietaed_res; // result

                    std::vector<std::string> parameters_abs =
                        this->get_param_vec((*it5));

                    // a) if *it5 is state-fluent
                    std::map<std::string, int>::iterator iter;
                    for (iter = (*(this->stateVariableIndices_net)).begin();
                         iter != (*(this->stateVariableIndices_net)).end();
                         iter++) {
                        std::vector<std::string> parameters_dependt_inst =
                            this->get_param_vec(iter->first);
                        std::vector<std::string> parameters_dependt_abst =
                            this->get_abs_vec_of_dependent_varied(
                                iter->first, sf_, it4->first);

                        if ((this->get_variable_name(iter->first) ==
                             this->get_variable_name(*it5))) {
                            bool involved = this->get_instantieted_ones(
                                params_abstract, params_instance,
                                parameters_dependt_abst,
                                parameters_dependt_inst);

                            if (involved &&
                                !(this->check_if_is_AF(
                                    this->erasechar(iter->first)))) {
                                intantietaed_res.push_back(iter->first);

                                // std::cout << sf_ << "," << *it5 << ","
                                //           << iter->first << std::endl;
                            };
                        }
                    };

                    // b) if *it5 is nonfluent
                    std::map<std::string, double>::iterator itk;
                    for (itk = this->NonfluentsToValues_net->begin();
                         itk != this->NonfluentsToValues_net->end(); itk++) {
                        if (this->get_variable_name(itk->first) ==
                            this->get_variable_name(*it5)) {
                            // Todo
                            std::vector<std::string> parameters_dependt_inst =
                                this->get_param_vec(itk->first);
                            std::vector<std::string> parameters_dependt_abst =
                                this->get_abs_vec_of_dependent_varied(
                                    itk->first, sf_, it4->first);

                            bool involved = this->get_instantieted_ones(
                                params_abstract, params_instance,
                                parameters_dependt_abst,
                                parameters_dependt_inst);
                            if (involved) {
                                intantietaed_res.push_back(itk->first);

                                //    std::cout <<sf_ <<", "<< *it5
                                //    <<","<<itk->first <<std::endl;
                            };
                        };
                    };

                    if (intantietaed_res.size() > 0) {
                        dependent_quantifir_sf_vec.push_back(*it5);
                        used.insert(this->get_variable_name(*it5));

                        instantiated_quantifier_SFs_[*it5] = intantietaed_res;
                        this->add_input_size_subm(*it5, input_size_subm);
                        input_size_subm_zero += 1;
                    }
                }
                it5++;
            }
            // quantifier -> abstract SFs
            order_of_quantifier_[it4->first] = dependent_quantifir_sf_vec;
        }

        // // (3)remaining non-fluents(without any parameters)
        // std::vector<std::string> nonfluents_withoutParams =
        //     this->initialize_nonfluent_Noparams(input_size_subm);

        // (3) remaining state-fluents
        std::map<std::string, std::set<std::string>> remaining_SFs;

        this->get_remaining(remaining_SFs, input_size_subm,
                            input_size_subm_zero, used);

        // initilalize dependency information
        dependency_info new_info = dependency_info(
            sf_, dependent_SFs_instantiated, instantiated_quantifier_SFs_,
            order_of_quantifier_, remaining_SFs);
        this->sub_models_dependency_information_[sf_] = new_info;

        // initialize the state fluents layers (sub models)

        // if already initialized, no need to initialize
        bool not_initilialized = false;
        for (unsigned int i = 0; i < this->layer; ++i) {
            if (torch_sub_models[i].count(sf_name) == 0) {
                not_initilialized = true;
            }
        }

        if (not_initilialized) {
            for (unsigned int i = 0; i < this->layer; i++) {
                // sub model has 3 layers:
                //             x  ;   (int)(x+3)/2   ;   3
                std::vector<int> sub_topology;

                int in_size;
                if (i == 0) {
                    in_size = input_size_subm_zero;
                } else {
                    in_size = input_size_subm;
                };
                int hidden_num = (int)((in_size + output_size_subM) * 3) / 4;
                int hidden_num2 = (int)((hidden_num + output_size_subM) / 2);
                sub_topology.push_back(in_size);
                sub_topology.push_back(hidden_num);
                sub_topology.push_back(hidden_num);
                sub_topology.push_back(hidden_num2);
                sub_topology.push_back(output_size_subM);
                assert(sub_topology.size() == submodule_size);
                // sub_model newsub = sub_model();
                auto newsub = std::make_shared<sub_model>(submodule_size);
                newsub->initialize(sub_topology);
                std::string mod_name = sf_name + std::to_string(i);
                newsub =
                    torch::nn::Module::register_module(mod_name, newsub); // ??
                torch_sub_models[i][sf_name] = *newsub;

                cout << to_string(in_size) << "layer" << to_string(i) << "//"
                     << mod_name << endl;
            }
        }
    }
}

bool Net::get_instantieted_ones(
    std::vector<std::string>& params_abstract,
    std::vector<std::string>& params_instance,
    std::vector<std::string>& parameters_dependt_abst,
    std::vector<std::string>& parameters_dependt_inst) {
    bool involved = true;
    int involved_num = 0;
    for (unsigned int ii = 0; ii < params_abstract.size(); ++ii) {
        for (unsigned int i3 = 0; i3 < parameters_dependt_abst.size(); ++i3) {
            // not(Left implies Right)
            if (!((!(params_abstract[ii] == parameters_dependt_abst[i3])) or
                  (params_instance[ii] == parameters_dependt_inst[i3]))) {
                involved = false;
            } else if ((params_abstract[ii] == parameters_dependt_abst[i3]) and
                       (params_instance[ii] == parameters_dependt_inst[i3])) {
                involved_num += 1;
            }
        }
    };
    if (involved_num == 0) {
        involved = false;
    };
    return involved;
}

void Net::add_input_size_subm(std::string name, int& input_size_subm) {
    // used to record the size
    bool sett = false;
    std::map<std::string, int>::iterator itqq;
    for (itqq = this->stateVariableIndices_net->begin();
         itqq != this->stateVariableIndices_net->end(); itqq++) {
        if (this->get_variable_name(name) ==
            this->get_variable_name(itqq->first)) {
            input_size_subm += this->output_size_subM;
            if (!sett) {
                sett = true;
            } else {
                SystemUtils::abort("Error.set twice.");
            }
            break;
        }
    };
    std::map<std::string, double>::iterator itqq2;
    for (itqq2 = this->NonfluentsToValues_net->begin();
         itqq2 != this->NonfluentsToValues_net->end(); itqq2++) {
        if ((this->get_variable_name(name) ==
             this->get_variable_name(itqq2->first)) &&
            !sett) {
            input_size_subm += 1;
            sett = true;
            break;
        }
    }
    if (!sett) {
        SystemUtils::abort("Error.not set .");
    }
    // used to record the size ----end
}

void Net::get_tmp_map(
    std::map<std::string, std::vector<std::string>>* tmp_map) {
    // get sf-> vector of its dependents sf

    std::map<std::string, std::set<std::string>>::iterator it1;
    for (it1 = this->sf2FixedSize_sfs_net->begin();
         it1 != this->sf2FixedSize_sfs_net->end(); it1++) {
        std::set<std::string> d_sf_set = it1->second;
        std::vector<std::string> dvec;
        std::set<std::string>::iterator iit2;

        for (iit2 = d_sf_set.begin(); iit2 != d_sf_set.end(); iit2++) {
            if (iit2->find(':') == std::string::npos) {
                dvec.push_back(*iit2);
            }
        }
        (*tmp_map)[it1->first] = dvec;
    }

    std::map<std::string,
             std::map<std::string, std::set<std::string>>>::iterator it2;
    for (it2 = this->sf2VariedSize_sfs_net->begin();
         it2 != this->sf2VariedSize_sfs_net->end(); it2++) {
        std::vector<std::string> dvec;
        if (tmp_map->count(it2->first) > 0) {
            dvec = (*tmp_map)[it2->first];
        }
        std::map<std::string, std::set<std::string>>::iterator it3;
        for (it3 = it2->second.begin(); it3 != it2->second.end(); it3++) {
            std::set<std::string>::iterator it4;
            for (it4 = it3->second.begin(); it4 != it3->second.end(); it4++) {
                if ((*it4).find(':') == std::string::npos) {
                    dvec.push_back(*it4);
                }
            }
        }
        (*tmp_map)[it2->first] = dvec;
    };
}

void Net::initialize_action_subMod(int submodule_size) {
    // get the types of actions
    std::vector<std::string> action_types_;
    for (unsigned int i = 0; i < this->action_f_net.size(); ++i) {
        std::string action_type =
            this->erasechar(this->get_variable_name((this->action_f_net)[i]));
        action_types_.push_back(action_type);
    }

    // get sf-> vector of its dependents sf
    std::map<std::string, std::vector<std::string>> tmp_map;
    this->get_tmp_map(&tmp_map);

    // abstract action type a -> ( abstract a ,dependent SFs/NFs for that)
    // abstract a
    std::map<std::string,
             std::vector<std::pair<std::string, std::vector<std::string>>>>
        map1;

    //  abstract action type a -> used sf type
    std::map<std::string, std::set<std::string>> used;

    for (int i = 0; i < action_types_.size(); i++) {
        std::vector<std::pair<std::string, std::vector<std::string>>> mypairs;
        std::set<std::string> used2;
        // if share any object, then add it.

        std::map<std::string, std::vector<std::string>>::iterator it1;
        for (it1 = tmp_map.begin(); it1 != tmp_map.end(); it1++) {
            std::vector<std::string> actionsMentioned;
            for (int j = 0; j < (it1->second).size(); j++) {
                std::string dname = (it1->second)[j];
                if (this->get_variable_name(dname) == action_types_[i]) {
                    actionsMentioned.push_back(dname);
                }
            }
            if (actionsMentioned.size() > 0) {
                for (int p = 0; p < actionsMentioned.size(); p++) {
                    std::vector<std::string> p_denpendent;

                    for (int j = 0; j < (it1->second).size(); j++) {
                        if (this->vec_is_intersect(
                                this->get_param_vec((it1->second)[j]),
                                this->get_param_vec(actionsMentioned[p]))) {
                            if (!this->check_if_is_AF((it1->second)[j])) {
                                p_denpendent.push_back((it1->second)[j]);
                                used2.insert(
                                    this->get_variable_name((it1->second)[j]));
                            }
                        };
                    }
                    if (this->vec_is_intersect(
                            this->get_param_vec(it1->first),
                            this->get_param_vec(actionsMentioned[p]))) {
                        p_denpendent.push_back(it1->first);
                        used2.insert(this->get_variable_name(it1->first));
                    };
                    if (p_denpendent.size() > 0) {
                        std::pair<std::string, std::vector<std::string>>
                            mypair =
                                make_pair(actionsMentioned[p], p_denpendent);
                        mypairs.push_back(mypair);
                    }
                }
            }
        }
        map1[action_types_[i]] = mypairs;
        // cout << "lllppp" << mypairs.size() << endl;
        used[action_types_[i]] = used2;
    };

    // get the dependency informations
    for (unsigned int i = 0; i < this->action_f_net.size(); ++i) {
        int input_size_subm = 0;
        std::string af_name = this->erasechar(this->action_f_net[i]);
        std::string af_type = this->get_variable_name(af_name);
        // abstract_Sf -> vector of instantiated sf  (same type but with
        // different instantiated
        std::map<std::string, std::vector<std::string>>
            instantiated_quantifier_SFs_subM;

        // quantifier -> vector of abstract dependent fluents
        std::map<std::string, std::vector<std::string>>
            order_of_quantifier_subM;
        std::vector<std::string> absones;

        std::vector<std::pair<std::string, std::vector<std::string>>> mypairs =
            map1[af_type];

        for (int j = 0; j < mypairs.size(); j++) {
            std::string abs_action = mypairs[j].first;
            std::vector<std::string> abs_sfs = mypairs[j].second;

            std::vector<std::string> params_abstract =
                this->get_param_vec(abs_action);
            std::vector<std::string> params_instance =
                this->get_param_vec(af_name);

            for (int j2 = 0; j2 < abs_sfs.size(); j2++) {
                std::vector<std::string> instantiatedOnes;
                std::vector<std::string> parameters_dependt_abst =
                    this->get_param_vec(abs_sfs[j2]);

                bool isSF = false;
                //(1)sf
                std::map<std::string, int>::iterator it3;
                for (it3 = this->stateVariableIndices_net->begin();
                     it3 != this->stateVariableIndices_net->end(); it3++) {
                    std::string sfName = this->erasechar(it3->first);
                    if (this->get_variable_name(sfName) ==
                            this->get_variable_name(abs_sfs[j2]) &&
                        !(this->check_if_is_AF(sfName))) {
                        std::vector<std::string> parameters_dependt_inst =
                            this->get_param_vec(sfName);
                        bool involved = this->get_instantieted_ones(
                            params_abstract, params_instance,
                            parameters_dependt_abst, parameters_dependt_inst);

                        if (involved) {
                            instantiatedOnes.push_back(sfName);
                            isSF = true;
                        }
                    }
                }
                //(2)nfs
                if (!isSF) {
                    std::map<std::string, double>::iterator itk;
                    for (itk = this->NonfluentsToValues_net->begin();
                         itk != this->NonfluentsToValues_net->end(); itk++) {
                        std::string NfName = this->erasechar(itk->first);
                        if (this->get_variable_name(abs_sfs[j2]) ==
                            this->get_variable_name(NfName)) {
                            std::vector<std::string> parameters_dependt_inst =
                                this->get_param_vec(NfName);
                            bool involved = this->get_instantieted_ones(
                                params_abstract, params_instance,
                                parameters_dependt_abst,
                                parameters_dependt_inst);
                            if (involved) {
                                instantiatedOnes.push_back(NfName);
                            };
                        };
                    };
                }
                if (instantiatedOnes.size() == 0) {
                    cout << "instantiatedOnes.size()==0"
                         << ", depednt: " << abs_sfs[j2]
                         << " ,main a:" << abs_action << endl;
                }
                if (instantiated_quantifier_SFs_subM.count(abs_sfs[j2]) == 0) {
                    absones.push_back(abs_sfs[j2]);
                    instantiated_quantifier_SFs_subM[abs_sfs[j2]] =
                        instantiatedOnes;
                } else {
                    cout << "second abs name encountered" << endl;
                    int m = 2;
                    while (instantiated_quantifier_SFs_subM.count(
                               std::to_string(m) + abs_sfs[j2]) != 0) {
                        m += 1;
                    }
                    std::string second_name = std::to_string(m) + abs_sfs[j2];
                    absones.push_back(second_name);
                    instantiated_quantifier_SFs_subM[second_name] =
                        instantiatedOnes;
                }

                input_size_subm += (isSF) ? this->output_size_subM : 1;
            }
        }

        // max-pooling over remaining
        int input_size_subm_zero;
        std::map<std::string, std::set<std::string>> remaining_SFs;
        this->get_remaining(remaining_SFs, input_size_subm,
                            input_size_subm_zero, used[af_type]);

        order_of_quantifier_subM["action"] = absones;

        std::vector<std::string> dependent_SFs_instantiated;
        // initialize dependency information
        dependency_info new_info =
            dependency_info(af_name, dependent_SFs_instantiated,
                            instantiated_quantifier_SFs_subM,
                            order_of_quantifier_subM, remaining_SFs);
        this->sub_models_dependency_information_[af_name] = new_info;

        // if already initialized, no need to initialize
        if (torch_sub_models[this->layer].count(af_type) != 0) {
            continue;
        }
        // not_initilialized
        std::vector<int> sub_topology;
        int hidden_num = (int)(input_size_subm + 5) / 2;
        sub_topology.push_back(input_size_subm);
        sub_topology.push_back(input_size_subm);
        sub_topology.push_back(hidden_num);
        sub_topology.push_back(hidden_num);
        sub_topology.push_back(5);
        sub_topology.push_back(4);
        sub_topology.push_back(1);
        assert(sub_topology.size() == submodule_size);
        // struct sub_model newsub = sub_model();
        auto newsub = std::make_shared<sub_model>(submodule_size);
        newsub->initialize(sub_topology);
        newsub = torch::nn::Module::register_module(af_type, newsub); // ??
        torch_sub_models[this->layer][af_type] = *newsub;

        std::cout << "-***--------action type:" << af_type << std::endl;
    }

    // no-op initialize
    std::string af_type = "no-op";
    int input_size_subm = 0;
    int input_size_subm_zero = 0;
    std::vector<std::string> dependent_SFs_instantiated;
    std::map<std::string, std::vector<std::string>> order_of_quantifier_subM;
    std::vector<std::string> absones;
    order_of_quantifier_subM["action"] = absones;
    std::map<std::string, std::vector<std::string>>
        instantiated_quantifier_SFs_subM;
    std::map<std::string, std::set<std::string>> remaining_SFs;
    std::set<std::string> emptyUsed;
    this->get_remaining(remaining_SFs, input_size_subm, input_size_subm_zero,
                        emptyUsed);
    dependency_info new_info = dependency_info(
        af_type, dependent_SFs_instantiated, instantiated_quantifier_SFs_subM,
        order_of_quantifier_subM, remaining_SFs);
    this->sub_models_dependency_information_[af_type] = new_info;
    std::vector<int> sub_topology;
    int hidden_num = (int)(input_size_subm + 5) / 2;
    sub_topology.push_back(input_size_subm);
    sub_topology.push_back(input_size_subm);
    sub_topology.push_back(hidden_num);
    sub_topology.push_back(hidden_num);
    sub_topology.push_back(5);
    sub_topology.push_back(4);
    sub_topology.push_back(1);
    assert(sub_topology.size() == submodule_size);
    auto newsub = std::make_shared<sub_model>(submodule_size);
    newsub->initialize(sub_topology);
    newsub = torch::nn::Module::register_module(af_type, newsub); // ??
    torch_sub_models[this->layer][af_type] = *newsub;
    std::cout << "-***--------action type:" << af_type << std::endl;
}

void Net::get_remaining(
    std::map<std::string, std::set<std::string>>& remaining_SFs,
    int& input_size_subm, int& input_size_subm_zero,
    std::set<std::string>& used) {
    //(a) remaining SFs

    std::map<std::string, std::set<std::string>>::iterator it1;
    for (it1 = this->sf2FixedSize_sfs_net->begin();
         it1 != this->sf2FixedSize_sfs_net->end(); it1++) {
        if (used.count(this->get_variable_name(it1->first)) == 0) {
            std::set<std::string> instantiated;
            std::map<std::string, int>::iterator it6;
            for (it6 = this->stateVariableIndices_net->begin();
                 it6 != this->stateVariableIndices_net->end(); it6++) {
                if (this->get_variable_name(it6->first) ==
                    this->get_variable_name(it1->first)) {
                    instantiated.insert(it6->first);
                }
            }
            if (instantiated.size() == 0) {
                cout << "should be NF in remaining SFs  : " << it1->first
                     << endl;
            }
            remaining_SFs[this->erasechar(it1->first)] = instantiated;
            input_size_subm += this->output_size_subM;
            input_size_subm_zero += 1;
        }
    }
    //(b) remaining nonFluents
    std::set<std::string>::iterator it6 = this->Nonfluents_types.begin();
    while (it6 != this->Nonfluents_types.end()) {
        if (used.count(*it6) == 0) {
            std::set<std::string> instantiated;
            std::map<std::string, double>::iterator itk;
            for (itk = this->NonfluentsToValues_net->begin();
                 itk != this->NonfluentsToValues_net->end(); itk++) {
                if (this->get_variable_name(itk->first) == *it6) {
                    instantiated.insert(itk->first);
                }
            }
            if (instantiated.size() == 0) {
                cout << "instantiated.size()==0 in remaining SFs : " << *it6
                     << endl;
            }
            remaining_SFs[*it6] = instantiated;
            input_size_subm += 1;
            input_size_subm_zero += 1;
        }
        it6++;
    }
}

std::vector<std::string> Net::get_abs_vec_of_dependent_varied(
    std::string instantiated_dependent_sf, std::string main_sf,
    std::string quantifier) {
    // input : an instantiated state-fluent
    // output : the abstract parameters of that state fluent in quantifier's
    // CPFs get the abstact parameters of instantiated_dependent_sf that in
    // the abs_sf's CPF
    std::vector<std::string> res;
    std::map<std::string,
             std::map<std::string, std::set<std::string>>>::iterator it1;
    for (it1 = this->sf2VariedSize_sfs_net->begin();
         it1 != this->sf2VariedSize_sfs_net->end(); it1++) {
        if (this->get_variable_name(it1->first) ==
            this->get_variable_name(main_sf)) {
            std::set<std::string> abs_set =
                (*(this->sf2VariedSize_sfs_net))[it1->first][quantifier];

            std::set<std::string>::iterator it2;
            for (it2 = abs_set.begin(); it2 != abs_set.end(); it2++) {
                if (this->get_variable_name(*it2) ==
                    this->get_variable_name(instantiated_dependent_sf)) {
                    res = this->get_param_vec(*it2);
                    return res;
                }
            }
        }
    }
    return res;
}

torch::Tensor Net::forward_helper() {
    std::array<std::string, 4> order_quantifier = {"Existential", "Universal",
                                                   "product", "sum"};
    // remaining layers

    for (int i = 0; i < this->layer; i++) {
        std::map<std::string, torch::Tensor> layer_Mi;
        std::map<std::string, int>::iterator tt;

        for (tt = this->stateVariableIndices_net->begin();
             tt != this->stateVariableIndices_net->end(); tt++) {
            dependency_info* info =
                &(sub_models_dependency_information_[tt->first]);

            sub_model* subm =
                &(torch_sub_models[i][this->get_variable_name(tt->first)]);

            // Order:  (1) dependent_SFs
            //         (2) "Existential" -> "Universal" -> "product" ->
            //         "sum", then according to order_of_quantifier_subM
            //         (3) remaining_SFs
            torch::Tensor in1;
            bool set = false;
            //(1)
            for (unsigned int j = 0; j < info->dependent_SFs.size(); j++) {
                std::string s1 = (info->dependent_SFs)[j];
                this->build_input_helper(set, s1, in1, i, "");
            }

            //(2)
            for (unsigned int j = 0; j < order_quantifier.size(); j++) {
                std::vector<std::string> abs_Sfs =
                    (info->order_of_quantifier_subM)[order_quantifier[j]];
                for (unsigned int j2 = 0; j2 < abs_Sfs.size(); j2++) {
                    std::vector<std::string> instantiated_ones =
                        (info->instantiated_quantifier_SFs_subM)[abs_Sfs[j2]];

                    torch::Tensor in11;
                    bool set2 = false;
                    // cout << "instantiated_ones"
                    //      << to_string(instantiated_ones.size()) <<
                    //      abs_Sfs[j2]
                    //      << tt->first << endl;

                    for (unsigned int j3 = 0; j3 < instantiated_ones.size();
                         j3++) {
                        std::string s1 = instantiated_ones[j3];

                        this->build_input_helper(set2, s1, in11, i, "max");
                    }

                    if (!set) {
                        in1 = in11;
                        if (set2) {
                            set = true;
                        }

                    } else {
                        if (set2) {
                            in1 = torch::cat({in1, in11}, 1);
                        } else {
                            cout << "problem" << endl;
                        }
                    }
                }
            };

            //(3)

            std::map<std::string, std::set<std::string>>::iterator itr;

            for (itr = info->remaining_SFs.begin();
                 itr != info->remaining_SFs.end(); itr++) {
                std::set<std::string> instantiated_ones = itr->second;

                torch::Tensor in11;
                bool set2 = false;

                std::set<std::string>::iterator it = instantiated_ones.begin();

                while (it != instantiated_ones.end()) {
                    std::string s1 = *it;

                    this->build_input_helper(set2, s1, in11, i, "max");
                    it++;
                }

                if (!set) {
                    in1 = in11;
                    if (set2) {
                        set = true;
                    }

                } else {
                    if (set2) {
                        in1 = torch::cat({in1, in11}, 1);
                    } else {
                        cout << "problem" << endl;
                    }
                }
            }

            // cout << "siz," << to_string(in1.sizes()[1]) << ", "
            //      << to_string(subm->layer_weights_sub[0].sizes()[0]) << ", "
            //      << "atlayer:" << std::to_string(i) << ", "
            //      << "name:" << tt->first << endl;

            torch::Tensor out = subm->forward(in1);
            layer_Mi[tt->first] = out;
            // if(i==2){
            //     cout<< tt->first<<": "<<out<<endl;
            // }

            // std::cout << "sf mod out:" << out << ", " << tt->first
            //           << ",at layer:" << std::to_string(i) << std::endl;
        };
        (this->layer_output)[i] = layer_Mi;
    };

    // action layer
    // Order:  (1)dependent_SFs
    //         (2) "action" to get the  order_of_quantifier_subM
    //         (3)  nonfluents_withoutParams_subM

    std::map<std::string, torch::Tensor> layer_Mi;
    for (int w = 0; w < this->action_f_net.size(); w++) {
        std::string action_ = (this->action_f_net)[w];
        dependency_info* info = &(sub_models_dependency_information_[action_]);
        sub_model* subm =
            &(torch_sub_models[this->layer][this->get_variable_name(action_)]);

        torch::Tensor in1;
        bool set = false;

        //(1)
        for (unsigned int j = 0; j < info->dependent_SFs.size(); j++) {
            std::string s1 = (info->dependent_SFs)[j];
            this->build_input_helper(set, s1, in1, this->layer, "");
        }

        //(2)
        std::vector<std::string> abs_Sfs =
            (info->order_of_quantifier_subM)["action"]; // To-do

        for (unsigned int j2 = 0; j2 < abs_Sfs.size(); j2++) {
            std::vector<std::string> instantiated_ones =
                (info->instantiated_quantifier_SFs_subM)[abs_Sfs[j2]];
            torch::Tensor in11;
            bool set2 = false;

            // cout << "instantiated_ones" <<
            // to_string(instantiated_ones.size())
            //      << abs_Sfs[j2] << action_ << endl;

            for (unsigned int j3 = 0; j3 < instantiated_ones.size(); j3++) {
                std::string s1 = instantiated_ones[j3];
                this->build_input_helper(set2, s1, in11, this->layer, "max");
            };

            // cout << "1111111111instantiated_ones.size()"
            //      << to_string(instantiated_ones.size()) << endl;

            if (!set) {
                in1 = in11;
                if (set2) {
                    set = true;
                }

            } else {
                if (set2) {
                    in1 = torch::cat({in1, in11}, 1);
                } else {
                    cout << "problem" << endl;
                }
            }
        }

        //(3)
        std::map<std::string, std::set<std::string>>::iterator itr;
        ;
        for (itr = info->remaining_SFs.begin();
             itr != info->remaining_SFs.end(); itr++) {
            std::set<std::string> instantiated_ones = itr->second;

            torch::Tensor in11;
            bool set2 = false;

            std::set<std::string>::iterator it = instantiated_ones.begin();

            while (it != instantiated_ones.end()) {
                std::string s1 = *it;
                this->build_input_helper(set2, s1, in11, this->layer, "max");
                it++;
            }
            // cout << "22222:-----" << to_string(instantiated_ones.size())
            //      << endl;
            if (!set) {
                in1 = in11;
                if (set2) {
                    set = true;
                }

            } else {
                if (set2) {
                    in1 = torch::cat({in1, in11}, 1);
                } else {
                    cout << "problem" << endl;
                }
            }
        }

        // cout << "test size:"
        //      << to_string(info->dependent_SFs.size() + abs_Sfs.size() +
        //                   info->remaining_SFs.size())
        //      << endl;
        torch::Tensor out = subm->forward(in1);
        // std::cout << "output action subm:" << out << "name:" << action_
        //           << std::endl;
        layer_Mi[action_] = out;
    };
    (this->layer_output)[this->layer] = layer_Mi;

    // forward for no-op

    std::string action_ = "no-op";
    dependency_info* info = &(sub_models_dependency_information_[action_]);
    sub_model* noopsubm = &(torch_sub_models[this->layer][action_]);

    torch::Tensor in1;
    bool set = false;

    //(3) no-op max over all types
    std::map<std::string, std::set<std::string>>::iterator itr;
    ;
    for (itr = info->remaining_SFs.begin(); itr != info->remaining_SFs.end();
         itr++) {
        std::set<std::string> instantiated_ones = itr->second;

        torch::Tensor in11;
        bool set2 = false;

        std::set<std::string>::iterator it = instantiated_ones.begin();

        while (it != instantiated_ones.end()) {
            std::string s1 = *it;
            this->build_input_helper(set2, s1, in11, this->layer, "max");
            it++;
        }
        // cout << "22222:-----" << to_string(instantiated_ones.size())
        //      << endl;
        if (!set) {
            in1 = in11;
            if (set2) {
                set = true;
            }

        } else {
            if (set2) {
                in1 = torch::cat({in1, in11}, 1);
            } else {
                cout << "problem" << endl;
            }
        }
    }

    // cout << "test size:"
    //      << to_string(info->dependent_SFs.size() + abs_Sfs.size() +
    //                   info->remaining_SFs.size())
    //      << endl;
    torch::Tensor no_op_out = noopsubm->forward(in1);
    // std::cout << "output action subm:" << no_op_out << "name:" << action_
    //           << std::endl;

    // torch::Tensor res = torch::zeros({1, 1}, options);
    torch::Tensor res = no_op_out;
    for (int k = 1; k < this->res_order.size(); k++) {
        res = torch::cat(
            {res, (this->layer_output)[this->layer][(this->res_order)[k]]}, 1);
    }

    // return torch::softmax(res, 1);
    // std::cout << "res" << res << std::endl;
    return res;
}

torch::Tensor Net::forward(
    std::map<std::string, torch::Tensor> const& layer_Minus1) {
    if (!value_set) {
        SystemUtils::abort("Error: not yet initialize the model.");
    }
    this->layer_output[-1] = layer_Minus1;
    //---------  -1 layer finished
    //----------------------------------------//
    return this->forward_helper();
    torch::Tensor res;
    return res;
}

torch::Tensor Net::forward(std::vector<double> const& training_data) {
    if (!value_set) {
        SystemUtils::abort("Error: not yet initialize the model.");
    }
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    // used to store the output of each layer. sub_model's name ->
    // sub_model's output
    std::map<std::string, torch::Tensor> layer_M1;

    //---------  -1 layer, input value to
    // tensor----------------------------------------//
    std::map<std::string, int>::iterator it;
    for (it = this->stateVariableIndices_net->begin();
         it != this->stateVariableIndices_net->end(); it++) {
        torch::Tensor tensor_ =
            torch::full({1, 1}, training_data[it->second], options);
        layer_M1[it->first] = tensor_;
    };

    std::map<std::string, double>::iterator itsd;
    for (itsd = this->NonfluentsToValues_net->begin();
         itsd != this->NonfluentsToValues_net->end(); itsd++) {
        std::string nf = itsd->first;
        double nf_val = itsd->second;
        torch::Tensor tensor_ = torch::full({1, 1}, nf_val, options);
        layer_M1[nf] = tensor_;
    };
    this->layer_output[-1] = layer_M1;
    //---------  -1 layer finished
    //----------------------------------------//

    return this->forward_helper();
}

void Net::build_input_helper(bool& set, std::string& s1, torch::Tensor& in1,
                             int i, std::string pooling_type) {
    if ((*(this->NonfluentsToValues_net)).count(s1) == 0) {
        // state fluents
        this->check_layer_output_exist(s1, i - 1);
        if (pooling_type == "max") {
            in1 = (set) ? torch::max(in1, this->layer_output[i - 1][s1])
                        : this->layer_output[i - 1][s1];
        } else {
            in1 = (set) ? torch::cat({in1, this->layer_output[i - 1][s1]}, 1)
                        : this->layer_output[i - 1][s1];
        }

    } else {
        // nonfluents
        this->check_layer_output_exist(s1, -1);
        if (pooling_type == "max") {
            in1 = (set) ? in1 = torch::max(in1, this->layer_output[-1][s1])
                        : in1 = this->layer_output[-1][s1];
        } else {
            in1 = (set) ? torch::cat({in1, this->layer_output[-1][s1]}, 1)
                        : this->layer_output[-1][s1];
        };
    }
    set = true;
}

std::vector<std::string> Net::get_param_vec(std::string str) {
    str = this->erasechar(str);
    std::vector<std::string> res;
    size_t cutPos = str.find('(');
    std::string allparam = str.substr(cutPos + 1);
    allparam = allparam.substr(0, allparam.length() - 1);
    StringUtils::split(allparam, res, ",");
    return res;
}

std::string Net::get_variable_name(std::string str) {
    str = erasechar(str);
    size_t cutPos = str.find('(');
    return str.substr(0, cutPos);
}

std::vector<long double> Net::tensor2vector_z(torch::Tensor in) {
    in = torch::softmax(in, 1);
    cout << "softmax" << in << endl;
    std::vector<long double> res;
    for (size_t i = 0; i < in.size(1); i++) {
        res.push_back((long double)in[0][i].item<double>());
        // std::cout << "tensor2vector_z"
        //           << "size:" << std::to_string(in.size(1))
        //           << (double)in[0][i].item<double>() << std::endl;
    }
    return res;
}

std::string Net::erasechar(std::string str) {
    std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
    str.erase(end_pos, str.end());
    return str;
}

bool Net::check_layer_output_exist(std::string s1, int layer_num) {
    if (this->layer_output[layer_num].count(s1) == 0) {
        std::string ss =
            "error:" + s1 + ",at layer:" + std::to_string(layer_num);
        std::cout << ss << std::endl;
        SystemUtils::abort(ss);
        return false;
    }
    return true;
}

bool Net::vec_is_intersect(std::vector<std::string> v1,
                           std::vector<std::string> v2) {
    bool res = false;
    for (size_t i = 0; i < v1.size(); i++) {
        for (size_t i2 = 0; i2 < v2.size(); i2++) {
            if (v1[i] == v2[i2]) {
                res = true;
            }
        }
    }
    return res;
}
bool Net::check_if_is_AF(std::string str) {
    bool res = false;
    for (unsigned int i = 0; i < (this->action_f_net.size()); ++i) {
        if (this->get_variable_name(str) ==
            this->get_variable_name(this->action_f_net[i])) {
            res = true;
        }
    }
    return res;
}

torch::Tensor Net::forward() {
    if (!state_pointer_set) {
        SystemUtils::abort("state_pointer_set not set!");
    }
    // std::cout << this->parameters() << std::endl;
    std::vector<double> current_state_ = *state_net;
    cout << "curr state vec: ";
    for (int i = 0; i < current_state_.size(); i++) {
        cout << to_string(current_state_[i]);
    }
    cout << endl;
    return forward(current_state_);
}