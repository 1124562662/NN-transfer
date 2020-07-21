#ifndef dependency_H
#define dependency_H

#include "Variables.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"

#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <vector>
using namespace std;

class dependency {
public:
    dependency(){};
};

class Action_module_dependency : public dependency {
public:
    Action_module_dependency(){};
    Action_module_dependency(
        InstantiatedVariable Ins_action,
        vector<AbstractRestrictedVariable> dependent_abs_fluents,
        vector<int> Inputs_types,
        map<string, vector<InstantiatedVariable>> dependent_ins_fluents)
        : Ins_action(Ins_action),
          dependent_abs_fluents(dependent_abs_fluents),
          Inputs_types(Inputs_types),
          dependent_ins_fluents(dependent_ins_fluents){};

    InstantiatedVariable Ins_action;
    vector<AbstractRestrictedVariable> dependent_abs_fluents;

   // 1 ->  no aggregation needed
    // 2 -> aggregation needed
    vector<int> Inputs_types;

    // ID of AbstractRestrictedVariable -> instantiated ones
    map<string, vector<InstantiatedVariable>> dependent_ins_fluents;
};

class Fluent_descriptor_dependency : public dependency {
public:
    Fluent_descriptor_dependency(){};
    Fluent_descriptor_dependency(
        InstantiatedVariable& mainAction, InstantiatedVariable& mainInsFluent,
        AbstractRestrictedVariable& mainAbsFluent,
        vector<AbstractRestrictedVariable>& descriptor_dependent_abs_fluents,
        map<string, vector<InstantiatedVariable>>&
            descriptor_dependent_ins_fluents,
        vector<int>& Inputs_types)
        : mainAction(mainAction),
          mainInsFluent(mainInsFluent),
          mainAbsFluent(mainAbsFluent),Inputs_types(Inputs_types),
          descriptor_dependent_abs_fluents(descriptor_dependent_abs_fluents),
          descriptor_dependent_ins_fluents(descriptor_dependent_ins_fluents)
          {};

    InstantiatedVariable mainAction;
    InstantiatedVariable mainInsFluent;
    AbstractRestrictedVariable mainAbsFluent;

    // 1 ->  no aggregation needed
    // 2 -> aggregation needed
    vector<int> Inputs_types;

    vector<AbstractRestrictedVariable> descriptor_dependent_abs_fluents;
    // ID of AbstractRestrictedVariable -> instantiated ones
    map<string, vector<InstantiatedVariable>> descriptor_dependent_ins_fluents;
};

#endif
