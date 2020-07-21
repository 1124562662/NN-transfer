
#include "AttentionNet.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <set>
#include <string>

using namespace std;

void AttentionNet::trainnet_() {
    cout << "training starts............." << endl;

    const int iter_num = 1000;

    const int start_point = 1;

    const int num_instances = 3;

    const int attention_layer_output_size = 3;

    const int batch_size_ = 220;

    const double learning_rate = 0.01;

    const double LRrate = 0.9;

    const std::string domainFolder = "elevator/";

    map<int, AttentionNet> modules;
    map<int, vector<vector<double>>> data_s;
    map<int, vector<double>> label_s;

    for (int y = start_point; y < num_instances + start_point; y++) {
        string folder = domainFolder + "instance" + std::to_string(y) + "/";

        AttentionNet tmp =
            AttentionNet(folder, attention_layer_output_size, false, "", false);
        modules[y] = tmp;

        if (y > start_point) {
            AttentionNet::check_domain_schema(
                tmp.StateFluentsSchema, tmp.NonFluentsSchema,
                tmp.ActionFluentsSchema, tmp.action_modules_sizes,
                tmp.fluent_descriptors_sizes, modules[y - 1].StateFluentsSchema,
                modules[y - 1].NonFluentsSchema,
                modules[y - 1].ActionFluentsSchema,
                modules[y - 1].action_modules_sizes,
                modules[y - 1].fluent_descriptors_sizes);
        }

        string line;
        vector<vector<double>> data;
        vector<double> label;

        ifstream myfile(folder + "data.txt");
        int i = 0;
        if (myfile.is_open()) {
            while (getline(myfile, line)) {
                if (i % 2 == 0) {
                    vector<double> data_point;
                    for (unsigned int i = 0; i < line.length(); i++) {
                        char c = line[i];
                        int tmp = (int)c - 48;
                        if (tmp > 0) {
                            data_point.push_back(1.0);
                        } else {
                            data_point.push_back(0);
                        }
                    }
                    data.push_back(data_point);
                    data_point.clear();
                } else {
                    label.push_back((double)std::stoi(line));
                }
                i = i + 1;
            }
            myfile.close();
        }
        assert(label.size() == data.size());
        data_s[y] = data;
        label_s[y] = label;
    };

    auto trainnet = std::make_shared<AttentionNet>(
        domainFolder + "instance" + std::to_string(start_point) + "/",
        attention_layer_output_size, false, "", false);

    torch::optim::AdamOptions op = torch::optim::AdamOptions()
                                       .weight_decay(0.000000001)
                                       .lr(
                                           /*lr=*/learning_rate);

    torch::optim::Adam optimizer(trainnet->parameters(), op);

    for (int iter = 0; iter < iter_num; iter++) {
        optimizer.zero_grad();

        if (iter % 1000 == 999) {
            for (auto& group : optimizer.param_groups()) {
                if (group.has_options()) {
                    auto& options = static_cast<torch::optim::AdamOptions&>(
                        group.options());
                    options.lr(options.lr() * (LRrate));
                }
            }
        };

        torch::Tensor loss;
        for (int y = start_point; y < num_instances + start_point; y++) {
            trainnet->change_instance(modules[y]);

            torch::Tensor pred;
            torch::Tensor labs;
            bool set = false;
            for (int batchi = 0; batchi < batch_size_; batchi++) {
                int rndi = rand() % (data_s[y].size());

                torch::Tensor prediction = trainnet->forward(data_s[y][rndi]);

                if (at::isnan(prediction).any().item<bool>()) {
                    cout << "nan" << to_string(batchi) << endl;
                    continue;
                }
                if (batchi % 100 == 0) {
                    cout << "predict:" << prediction << endl;
                }
                pred = (set) ? torch::cat({pred, prediction}, 0) : prediction;

                torch::Tensor label_tor =
                    torch::full({1}, (int)label_s[y][rndi],
                                torch::TensorOptions().dtype(torch::kLong));

                labs = (set) ? torch::cat({labs, label_tor}, 0) : label_tor;
                set = true;
            }

            torch::Tensor loss_ =
                torch::nn::functional::cross_entropy(pred, labs);
            cout << "loss_" << to_string(y) << ",  " << loss_ << endl;

            if (y == start_point) {
                loss = loss_;
            } else {
                loss = torch::add(loss, loss_);
            };
        }
        cout << "loss" << to_string(iter) << ",  " << loss << endl;
        cout << "" << endl;
        cout << "" << endl;

        loss.backward();
        optimizer.step();
    }
}