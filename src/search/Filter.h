#ifndef FILTER_H
#define FILTER_H
#include <iostream>
#include <set>
#include <string>
#include <vector>
using namespace std;

#include "Variables.h"
class filter {
public:
    filter(){};
    filter(vector<AbstractVariable>& absVars,
            vector< vector<pair<pair<int, int>, pair<int, int>>>>& restrictions)
        : absVars(absVars), restrictions(restrictions){};

    vector<AbstractVariable> absVars;
    // the i-th AbsVar's k-th parameter is unified with j-ith AbsVar's w-th
    // parameter vector of <<i,k>,<j,w>>
   vector< vector<pair<pair<int, int>, pair<int, int>>>> restrictions;

    vector<vector<InstantiatedVariable>> InstantiatedOnes;

private:
};
#endif
