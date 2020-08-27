#ifndef DOMIANGRAPH_H
#define DOMIANGRAPH_H
#include "Edges.h"
#include "Nodes.h"
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "Variables.h"

using namespace std;
class domian_graph {
public:
    domian_graph(){};

    void Set_sizes(int relation_node_output_size,
                   double relation_node_growing_speed, int time_step_K);

    map<string, domain_edge> DomainEdgeTypeMap;
    map<string, relation_type_vertex> Domian_relation_V_map;
    map<string, object_type_vertex> Domian_Object_V_map;
};
#endif
