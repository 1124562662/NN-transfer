#include "mln.h"
#include "utils/system_utils.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>

using namespace std;

myMLN::myMLN(vector<AbstractVariable>& vecVars) {
    set<string> objs;

    for (int i = 0; i < vecVars.size(); i++) {
        for (int j = 0; j < vecVars[i].patameters_types.size(); j++) {
            objs.insert(vecVars[i].patameters_types[j]);
        }
    }
    set<string>::iterator it = objs.begin();

    while (it != objs.end()) {
        vector<AbstractVariable> uniVars;
        vector<AbstractVariable> neighbours;

        for (int i = 0; i < vecVars.size(); i++) {
            if (vecVars[i].patameters_types.size() == 1) {
                if (vecVars[i].patameters_types[0] == *it) {
                    uniVars.push_back(vecVars[i]);
                }
            } else if (vecVars[i].patameters_types.size() > 1) {
                if (this->contains(*it, vecVars[i].patameters_types)) {
                    neighbours.push_back(vecVars[i]);
                }
            }
        }
        node n = node(*it, uniVars, neighbours);
        (this->nodes).push_back(n);
        (this->nodesmap)[*it] = n;
        it++;
    }
}

filter myMLN::perform_random_walk(int length, string walktype) {
    vector<AbstractVariable> absVars_F;
    // the i-th AbsVar's k-th parameter is unified with j-ith AbsVar's w-th
    // parameter vector of <<i,k>,<j,w>>
    vector<vector<pair<pair<int, int>, pair<int, int>>>> restrictions_F;
    // set<string> used_object_types;

    int start_node_idx = rand() % (nodes.size());
    node current_node = nodes[start_node_idx];

    int current_length = 0;
    if (walktype == "DFS") {
        this->DFS_helper(current_length, length, current_node, absVars_F,
                         restrictions_F);
    } else if (walktype == "BFS") {
        queue<Queue_elem> q;
        int current_length = 0;
        // int qidx = 0;
        // map<int, int> qidxMAP;

        pair<int, int> p1;
        if (current_node.uniVars.size() > 0) {
            p1 = make_pair(0, 0);
        } else {
            vector<int> idx;
            for (int i = 0;
                 i < current_node.neighbours[0].patameters_types.size(); i++) {
                if (current_node.neighbours[0].patameters_types[i] ==
                    current_node.object_type) {
                    idx.push_back(i);
                }
            }
            p1 = make_pair(0, idx[rand() % idx.size()]);
        }
        Queue_elem qm = Queue_elem(current_node.object_type, p1);
        q.push(qm);

        // int dequeue_num = 0;
        while (current_length <= length && (!q.empty())) {
            set<Queue_elem, Queue_elem_comp> frontier;

            // qidxMAP[dequeue_num] = current_length;
            // dequeue_num += 1;

            this->BFS_helper(current_length, q, absVars_F, restrictions_F,
                             frontier);

            while (!frontier.empty()) {
                int rnd = rand() % frontier.size();
                set<Queue_elem>::const_iterator it(frontier.begin());
                advance(it, rnd);
                // Queue_elem qm2 = Queue_elem(it->type_name, it->restriction);
                q.push(*it);
                frontier.erase(it);
            }
        }
    }
    filter res = filter(absVars_F, restrictions_F);
    return res;
}

void myMLN::BFS_helper(
    int& current_length, queue<Queue_elem>& q,
    vector<AbstractVariable>& absVars_F,
    vector<vector<pair<pair<int, int>, pair<int, int>>>>& restrictions_F,
    set<Queue_elem, Queue_elem_comp>& frontier) {
    Queue_elem q_elem = q.front();
    q.pop();
    node current_node = nodesmap[q_elem.type_name];

    pair<int, int> p1 = q_elem.restriction;

    for (int j = 0; j < current_node.uniVars.size(); j++) {
        vector<pair<pair<int, int>, pair<int, int>>> restriction_f;

        pair<int, int> p2 = make_pair(current_length, 0);
        assert(p1.first <= p2.first);
        pair<pair<int, int>, pair<int, int>> p3 = make_pair(p1, p2);
        restriction_f.push_back(p3);
        restrictions_F.push_back(restriction_f);
        absVars_F.push_back(current_node.uniVars[j]);
        current_length++;
    };

    for (int i = 0; i < current_node.neighbours.size(); i++) {
        vector<int> idx;
        for (int j = 0; j < current_node.neighbours[i].patameters_types.size();
             j++) {
            if (current_node.neighbours[i].patameters_types[j] ==
                current_node.object_type) {
                idx.push_back(j);
            }
        }

        pair<int, int> p2 = make_pair(current_length, idx[rand() % idx.size()]);
        assert(p1.first <= p2.first);
        pair<pair<int, int>, pair<int, int>> p3 = make_pair(p1, p2);
        vector<pair<pair<int, int>, pair<int, int>>> restriction_f;
        restriction_f.push_back(p3);
        restrictions_F.push_back(restriction_f);
        absVars_F.push_back(current_node.neighbours[i]);

        for (int j = 0; j < current_node.neighbours[i].patameters_types.size();
             j++) {
            if (current_node.neighbours[i].patameters_types[j] !=
                current_node.object_type) {
                string typ = current_node.neighbours[i].patameters_types[j];
                pair<int, int> p = make_pair(current_length, j);
                Queue_elem qm = Queue_elem(typ, p);
                frontier.insert(qm);
            }
        }
        
        current_length++;
    };
}

void myMLN::DFS_helper(
    int& current_length, int& length, node& current_node,
    vector<AbstractVariable>& absVars_F,
    vector<vector<pair<pair<int, int>, pair<int, int>>>>& restrictions_F) {
    if (current_length <= length) {
        for (int j = 0; j < current_node.uniVars.size(); j++) {
            vector<pair<pair<int, int>, pair<int, int>>> restriction_f;
            pair<int, int> p1;
            pair<int, int> p2;
            if (current_length == 0) {
                p1 = make_pair(current_length, 0);
                p2 = make_pair(current_length, 0);
            } else {
                p1 = make_pair(
                    current_length - 1,
                    get_index_obj(absVars_F, current_node.object_type));
                p2 = make_pair(current_length, 0);
            };
            assert(p1.first <= p2.first);
            pair<pair<int, int>, pair<int, int>> p3 = make_pair(p1, p2);
            restriction_f.push_back(p3);
            restrictions_F.push_back(restriction_f);
            absVars_F.push_back(current_node.uniVars[j]);
            current_length++;
        }

    } else {
        return;
    };

    if (current_length <= length) {
        AbstractVariable neighbour =
            current_node.neighbours[rand() % (current_node.neighbours.size())];

        vector<pair<pair<int, int>, pair<int, int>>> restriction_f;
        pair<int, int> p1;
        pair<int, int> p2;
        vector<AbstractVariable> tmp;
        tmp.push_back(neighbour);
        if (current_length == 0) {
            p1 = make_pair(current_length,
                           get_index_obj(tmp, current_node.object_type));
            p2 = make_pair(current_length,
                           get_index_obj(tmp, current_node.object_type));
        } else {
            p1 = make_pair(current_length - 1,
                           get_index_obj(absVars_F, current_node.object_type));
            p2 = make_pair(current_length,
                           get_index_obj(tmp, current_node.object_type));
        };
        assert(p1.first <= p2.first);
        pair<pair<int, int>, pair<int, int>> p3 = make_pair(p1, p2);
        restriction_f.push_back(p3);
        restrictions_F.push_back(restriction_f);
        absVars_F.push_back(neighbour);
        current_length++;

        vector<string> newObjs;
        for (int k = 0; k < neighbour.patameters_types.size(); k++) {
            newObjs.push_back(neighbour.patameters_types[k]);
        }
        current_node = nodesmap[newObjs[rand() % newObjs.size()]];
        this->DFS_helper(current_length, length, current_node, absVars_F,
                         restrictions_F);
    } else {
        return;
    };
    return;
}

int myMLN::get_index_obj(vector<AbstractVariable>& absVars_F, string objType) {
    assert(absVars_F.size() > 0);
    vector<int> res;
    for (int i = 0; i < absVars_F[absVars_F.size() - 1].patameters_types.size();
         i++) {
        string tmp = absVars_F[absVars_F.size() - 1].patameters_types[i];
        if (tmp == objType) {
            res.push_back(i);
        }
    }

    assert(res.size() > 0);
    if (res.size() == 0) {
        SystemUtils::abort("mmm.");
    }
    return res[rand() % res.size()];
}

bool myMLN::contains(string a, vector<string>& as) {
    for (int i = 0; i < as.size(); i++) {
        if (as[i] == a) {
            return true;
        }
    }
    return false;
}