#ifndef dependency_info_H
#define dependency_info_H

class dependency_info {
public:
    dependency_info(){};
    dependency_info(
        std::string name_2, std::vector<std::string> dependent_SFs2,
        std::map<std::string, std::vector<std::string>>
            instantiated_quantifier_SFs_subM2,
        std::map<std::string, std::vector<std::string>>
            order_of_quantifier_subM2,

        std::map<std::string, std::set<std::string>> remaining_SFs2) {
        this->name_ = name_2;
        this->dependent_SFs = dependent_SFs2;
        this->instantiated_quantifier_SFs_subM =
            instantiated_quantifier_SFs_subM2;
        this->order_of_quantifier_subM = order_of_quantifier_subM2;
        this->remaining_SFs = remaining_SFs2;
    };
    std::string name_;
    std::vector<std::string> dependent_SFs; // instantiated sfs

    // abstract_Sf -> vector of instantiated sf  (same type but with
    // different instantiated
    std::map<std::string, std::vector<std::string>>
        instantiated_quantifier_SFs_subM;


    // quantifier -> vector of abstract dependent fluents
    std::map<std::string, std::vector<std::string>> order_of_quantifier_subM;

    // nonfluents that do not have any parameters
    std::vector<std::string> nonfluents_withoutParams_subM;

    // remaining SFs that are not used by the previous ones
    std::map<std::string, std::set<std::string>> remaining_SFs;

};

#endif
