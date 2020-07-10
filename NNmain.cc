#include "prost_planner.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <vector>

using namespace std;
std::string erasechar(std::string str);
int ProstPlanner::trainnet_() {
    cout << "training starts............." << endl;
    static int iter_num = 100;

    static int inner_iter_num = 1000;

    static int num_instances = 1;
    auto trainnet = std::make_shared<ProstPlanner::Net>();
    trainnet->initialize_(true, "instance" + std::to_string(1) + "/", -1);
    std::map<int, std::map<std::string, ProstPlanner::sub_model>>
        torch_sub_models_share = trainnet->torch_sub_models;

    for (int iter = 0; iter < iter_num; iter++) {
        for (int y = 1; y < num_instances + 1; y++) {
            cout << "training for iteration:" + to_string(iter) +
                        " ,instance:" + to_string(y)
                 << endl;
                 
            string folder = "instance" + std::to_string(y) + "/";

            auto trainnet = std::make_shared<ProstPlanner::Net>();

            trainnet->initialize_(true, folder, -1);
            trainnet->torch_sub_models = torch_sub_models_share;

            // 1 data containers
            std::map<std::string, double> NonfluentsToValues;
            trainnet->NonfluentsToValues_net = &NonfluentsToValues;
            // key : string of state fluents with parameters
            // value : a vector of fixed-size dependent state fluents with
            // parametetrs
            std::map<std::string, std::set<std::string>> sf2FixedSize_sfs;
            trainnet->sf2FixedSize_sfs_net = &sf2FixedSize_sfs;

            // key : string of state fluents with parameters -> quantifiers
            // -> vector of parameters and dependent state fluents
            std::map<std::string, std::map<std::string, std::set<std::string>>>
                sf2VariedSize_sfs;
            trainnet->sf2VariedSize_sfs_net = &sf2VariedSize_sfs;

            // state fluent string -> {[object,type]}
            std::map<std::string, std::set<std::vector<std::string>>>
                sf_parameters_type_z;
            trainnet->sf_parameters_type_z_net = &sf_parameters_type_z;

            // 2 read parameters
            std::ifstream taskFile2;
            taskFile2.open(folder + "non-fluents-file");
            if (taskFile2.is_open()) {
                std::string line;
                while (getline(taskFile2, line)) {
                    std::string name_ = erasechar(line);
                    getline(taskFile2, line);
                    std::string value_ = line;
                    double value2 = std::stod(value_);
                    NonfluentsToValues[name_] = value2;
                    // std::cout<<
                    // std::to_string((this->NonfluentsToValues)[name_])<<std::endl;
                    // std::cout<< name_<<std::endl;
                }
                taskFile2.close();
            }

            std::ifstream taskFile22;
            taskFile22.open(folder + "CDFs_file_");
            if (taskFile22.is_open()) {
                std::map<std::string, std::set<std::string>> tmp_quantifier_;
                std::string current_sf_name_;
                std::string line;
                while (getline(taskFile22, line)) {
                    if (line == "ExistentialQuantification:") {
                        std::set<std::string> tmp_para_sfs;
                        getline(taskFile22, line);
                        std::string parameter_z = erasechar(line);
                        tmp_para_sfs.insert(parameter_z);
                        getline(taskFile22, line);
                        while (line == "Existential-fluent_name:") {
                            getline(taskFile22, line);
                            tmp_para_sfs.insert(erasechar(line));
                            getline(taskFile22, line);
                        }
                        tmp_quantifier_["Existential"] = tmp_para_sfs; // Todo
                    } else if (line == "UniversalQuantification:") {
                        std::set<std::string> tmp_para_sfs;
                        getline(taskFile22, line);
                        std::string parameter_z = erasechar(line);
                        tmp_para_sfs.insert(erasechar(line));
                        getline(taskFile22, line);
                        while (line == "Universal-fluent_name:") {
                            getline(taskFile22, line);
                            tmp_para_sfs.insert(erasechar(line));
                            getline(taskFile22, line);
                        }
                        tmp_quantifier_["Universal"] = tmp_para_sfs;

                    } else if (line == "Product:") {
                        std::set<std::string> tmp_para_sfs;
                        getline(taskFile22, line);
                        std::string parameter_z = erasechar(line);
                        tmp_para_sfs.insert(erasechar(line));
                        getline(taskFile22, line);
                        while (line == "prod-fluent_name:") {
                            getline(taskFile22, line);
                            tmp_para_sfs.insert(erasechar(line));
                            getline(taskFile22, line);
                        }
                        tmp_quantifier_["product"] = tmp_para_sfs;

                    } else if (line == "Sumation:") {
                        std::set<std::string> tmp_para_sfs;
                        getline(taskFile22, line);
                        std::string parameter_z = erasechar(line);
                        tmp_para_sfs.insert(erasechar(line));
                        getline(taskFile22, line);
                        while (line == "sum-fluent_name:") {
                            getline(taskFile22, line);
                            tmp_para_sfs.insert(erasechar(line));
                            getline(taskFile22, line);
                        }
                        tmp_quantifier_["sum"] = tmp_para_sfs;
                    } else if (line == "The sf:") {
                        getline(taskFile22, line);
                        current_sf_name_ = erasechar(line);
                        sf2VariedSize_sfs[current_sf_name_] = tmp_quantifier_;
                        tmp_quantifier_.clear();
                        getline(taskFile22, line);
                        while (line == "obj:") {
                            std::vector<std::string> obj_and_type;
                            getline(taskFile22, line);
                            obj_and_type.push_back(erasechar(line));
                            getline(taskFile22, line);
                            obj_and_type.push_back(erasechar(line));
                            getline(taskFile22, line);
                            if (sf_parameters_type_z.count(current_sf_name_) ==
                                0) {
                                std::set<std::vector<std::string>> tmp_pair_z;
                                tmp_pair_z.insert(obj_and_type);
                                sf_parameters_type_z[current_sf_name_] =
                                    tmp_pair_z;
                            } else {
                                sf_parameters_type_z[current_sf_name_].insert(
                                    obj_and_type);
                            }
                        }
                    } else if (line == "fix-fluent_name:") {
                        getline(taskFile22, line);
                        if (sf2FixedSize_sfs.count(
                                erasechar(current_sf_name_)) == 0) {
                            std::set<std::string> tmp_sfs;
                            tmp_sfs.insert(erasechar(line));
                            sf2FixedSize_sfs[current_sf_name_] = tmp_sfs;
                        } else {
                            sf2FixedSize_sfs[current_sf_name_].insert(
                                erasechar(line));
                        }
                    }
                }
                taskFile22.close();
            }

            // construct the input
            int i = 0;

            string line;
            vector<vector<double>> data;
            vector<vector<double>> label;
            static int input_size = 11;
            static int out_size = 5;
            ifstream myfile(folder + "data.txt");
            if (myfile.is_open()) {
                while (getline(myfile, line)) {
                    if (i % 2 == 0) {
                        vector<double> data_point(input_size, 0);
                        for (unsigned int i = 0; i < line.length(); i++) {
                            char c = line[i];
                            int tmp = (int)c - 48;
                            if (tmp > 0) {
                                data_point[i] = 1.0;
                            } else {
                                data_point[i] = -1.0;
                            }
                        }
                        data.push_back(data_point);
                        data_point.clear();

                    } else {
                        vector<double> label_point(out_size, 0);
                        for (unsigned int i = 0; i < line.length(); i++) {
                            char c = line[i];
                            int tmp = (int)c - 48;
                            label_point[tmp] = 1.0;
                        }
                        label.push_back(label_point);
                        label_point.clear();
                    }
                    i = i + 1;
                }
                myfile.close();
            }

            // inner interation
            torch::optim::SGD optimizer(trainnet->parameters(), /*lr=*/0.01);

            for (int epoch = 0; epoch < inner_iter_num; ++epoch) {
                optimizer.zero_grad();
                int rndi = rand() % data.size();
                torch::Tensor prediction = trainnet->forward(data[rndi]);
                torch::Tensor label_tor =
                    torch::zeros({1, (int)(label[rndi]).size()});
                int counter = 0;
                for (int i : (label[rndi])) {
                    label_tor[0][counter] = i;
                    counter++;
                }
                cout << "label" << label_tor << endl;
                cout << "prediction" << prediction << endl;

                torch::Tensor loss = torch::nll_loss(prediction, label_tor);
                if (epoch % 100 == 0) {
                    cout << "loss" << loss << endl;
                }

                loss.backward();
                // Update the parameters based on the calculated gradients.
                optimizer.step();
            }

            data.clear();
            label.clear();

            // save parameters
            torch_sub_models_share = trainnet->torch_sub_models;
        }
    }

    return 0;
}

std::string erasechar(std::string str) {
    std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
    str.erase(end_pos, str.end());
    return str;
}
