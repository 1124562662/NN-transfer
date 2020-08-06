#ifndef MLN_H
#define MLN_H

#include "Filter.h"
#include "Variables.h"
#include "assert.h"
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>
using namespace std;

class myMLN {
public:
    myMLN(){};
    myMLN(vector<AbstractVariable>& vecVars);

    struct node {
    public:
        node(){};
        node(string object_type, vector<AbstractVariable> uniVars,
             vector<AbstractVariable> neighbours)
            : object_type(object_type),
              uniVars(uniVars),
              neighbours(neighbours){};

        // ID
        string object_type;
        vector<AbstractVariable> uniVars;
        // not AbsVar that has only 1 object
        vector<AbstractVariable> neighbours;
        // vector<double> probs;
    };
    // object_type -> absVar name -> node
    // map<string, map<string, node>> graph;
    vector<node> nodes;
    // object_type ->  node
    map<string, node> nodesmap;

    filter perform_random_walk(int length, string walktype);

    struct Queue_elem {
        Queue_elem(){};
        Queue_elem(string type_name, pair<int, int> restriction)
            : type_name(type_name), restriction(restriction){};
        string type_name;
        pair<int, int> restriction;
    };

    struct Queue_elem_comp {
        bool operator()(const Queue_elem& q1, const Queue_elem& q) const {
            return q1.type_name + to_string(q1.restriction.first) +
                       to_string(q1.restriction.second) <
                   q.type_name + to_string(q.restriction.first) +
                       to_string(q.restriction.second);
        }
    };

    void updateMLN(filter& f, double attantionValue);

private:
    void buildMLN();
    void DFS_helper(
        int& current_length, int& length, node& current_node,
        vector<AbstractVariable>& absVars_F,
        vector<vector<pair<pair<int, int>, pair<int, int>>>>& restrictions_F);

    void BFS_helper(
        int& current_length, queue<Queue_elem>& q,
        vector<AbstractVariable>& absVars_F,
        vector<vector<pair<pair<int, int>, pair<int, int>>>>& restrictions_F,
        set<Queue_elem, Queue_elem_comp>& frontier);

    int get_index_obj(vector<AbstractVariable>& absVars_F, string objType);

    static bool contains(string a, vector<string>& as);
};
#endif
