#include "ipc_client.h"

#include "Net.h"
#include "parser.h"
#include "prost_planner.h"
#include "utils/base64.h"
#include "utils/string_utils.h"
#include "utils/strxml.h"
#include "utils/system_utils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <logger.h>
#include <netdb.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

using namespace std;

IPCClient::IPCClient(string _hostName, unsigned short _port,
                     string _parserOptions)
    : hostName(_hostName),
      port(_port),
      socket(-1),
      parserOptions(_parserOptions),
      numberOfRounds(-1),
      remainingTime(0) {}

// This destructor is required here to allow forward declaration of
// ProstPlanner in header because of usage of unique_ptr<ProstPlanner>
IPCClient::~IPCClient() = default;

void IPCClient::run(string const& instanceName, string& plannerDesc) {
    // Reset static members from possible earlier runs
    ProstPlanner::resetStaticMembers();
    // Init connection to the rddlsim server
    initConnection();

    // Request round
    initSession(instanceName, plannerDesc);

    vector<double> nextState(stateVariableIndices.size());

    // 4550 tmp
   // planner->mynet_z.state_net = &nextState;
    planner->mynet_z.set_state_pointer( &nextState);
    /// testtesttesttesttesttesttest
    // torch::Tensor rr = planner->mynet_z.forward();
    // std::cout << "forward result:" << std::endl;
    // std::cout << rr << std::endl;

    // std::cout << "fend" << std::endl;

    /// testtesttesttesttesttesttest
    // 4550 tmp

    double immediateReward = 0.0;

    // Main loop
    for (unsigned int currentRound = 0; currentRound < numberOfRounds;
         ++currentRound) {
        initRound(nextState, immediateReward);

        while (true) {
            planner->initStep(nextState, remainingTime);
            vector<string> nextActions = planner->plan();
            if (!submitAction(nextActions, nextState, immediateReward)) {
                break;
            }
            planner->finishStep(immediateReward);

            //         map<std::string, int>::iterator it;
            //    for (it = stateVariableIndices.begin(); it !=
            //    stateVariableIndices.end(); it++) {    std::vector<double>*
            //    stmp=planner->mynet_z.state_net;
            //         std::cout<<it->first<<" :: "<<
            //         to_string((*stmp)[it->second])<<std::endl;
            //    }
        }
    }

    // Get end of session message and print total result
    finishSession();

    // Close connection to the rddlsim server
    closeConnection();
}

/******************************************************************************
                               Server Communication
******************************************************************************/

void IPCClient::initConnection() {
    assert(socket == -1);
    try {
        socket = connectToServer();
        if (socket <= 0) {
            SystemUtils::abort("Error: couldn't connect to server.");
        }
    } catch (const exception& e) {
        SystemUtils::abort("Error: couldn't connect to server.");
    } catch (...) {
        SystemUtils::abort("Error: couldn't connect to server.");
    }
}

int IPCClient::connectToServer() {
    struct hostent* host = ::gethostbyname(hostName.c_str());
    if (!host) {
        return -1;
    }

    int res = ::socket(PF_INET, SOCK_STREAM, 0);
    if (res == -1) {
        return -1;
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr = *((struct in_addr*)host->h_addr);
    memset(&(addr.sin_zero), '\0', 8);

    if (::connect(res, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        return -1;
    }
    return res;
}

void IPCClient::closeConnection() {
    if (socket == -1) {
        SystemUtils::abort("Error: couldn't disconnect from server.");
    }
    close(socket);
}

/******************************************************************************
                     Session and rounds management
******************************************************************************/

void IPCClient::initSession(string const& instanceName, string& plannerDesc) {
    stringstream os;
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
       << "<session-request>"
       << "<problem-name>" << instanceName << "</problem-name>"
       << "<client-name>"
       << "prost"
       << "</client-name>"
       << "<input-language>rddl</input-language>"
       << "<no-header/>"
       << "</session-request>" << '\0';
    if (write(socket, os.str().c_str(), os.str().length()) == -1) {
        SystemUtils::abort("Error: writing to socket failed.");
    }

    const XMLNode* serverResponse = XMLNode::readNode(socket);

    if (!serverResponse) {
        SystemUtils::abort("Error: initializing session failed.");
    }

    string s;
    // If the task was not initialized, we have to read it from the server and
    // run the parser
    assert(SearchEngine::taskName.empty());
    if (!serverResponse->dissect("task", s)) {
        SystemUtils::abort(
            "Error: server response does not contain task description.");
    }

    s = decodeBase64(s);
    executeParser(s, &(this->action_f_z));
    if (!serverResponse->dissect("num-rounds", s)) {
        SystemUtils::abort("Error: server response insufficient.");
    }
    numberOfRounds = atoi(s.c_str());

    if (!serverResponse->dissect("time-allowed", s)) {
        SystemUtils::abort("Error: server response insufficient.");
    }
    remainingTime = atoi(s.c_str());

    delete serverResponse;
    // in c++ 14 we would use make_unique<ProstPlanner>
 Net::trainnet_();
//this->saving=true;
    planner = std::unique_ptr<ProstPlanner>(new ProstPlanner(plannerDesc));
    planner->initSession(numberOfRounds, remainingTime);

    // 4550 tmp

    std::map<std::string, std::set<std::string>>::iterator iii;
    std::cout << "oooojhdbwcwuhuc" << std::endl;
    for (iii = this->sf2FixedSize_sfs.begin();
         iii != this->sf2FixedSize_sfs.end(); iii++) {
        std::cout << iii->first << std::endl;
    }

    // remove ' '
    std::map<std::string, int> newsvi;
    std::map<std::string, int>::iterator it_;
    for (it_ = this->stateVariableIndices.begin();
         it_ != this->stateVariableIndices.end(); it_++) {
        newsvi[erasechar(it_->first)] = it_->second;
    }
    this->stateVariableIndices = newsvi;
    // end of removing

    std::vector<std::string> action_f_ipc;
    for (int i = 0; i < (this->action_f_z.size()); ++i) {
        std::stringstream stros;
        (this->action_f_z)[i]->print(stros);
        // std::cout << this->erasechar(stros.str()) << std::endl;
        action_f_ipc.push_back(this->erasechar(stros.str()));
    }
    // std::cout << std::to_string(this->action_f_z.size()) << std::endl;

    planner->mynet_z.initialize_(
      !(this->saving), "E1/", 3, &(this->NonfluentsToValues), &(this->sf2VariedSize_sfs),
        &(this->sf2FixedSize_sfs), &(this->stateVariableIndices),
        &(this->stateVariableValues), action_f_ipc, &(this->action_indices));

    //
    
    if (this->saving) {
        planner->mynet_z.save_parameters("instance/");
    }
}

void IPCClient::finishSession() {
    XMLNode const* sessionEndResponse = XMLNode::readNode(socket);

    if (sessionEndResponse->getName() != "session-end") {
        SystemUtils::abort("Error: session end message insufficient.");
    }
    string s;
    if (!sessionEndResponse->dissect("total-reward", s)) {
        SystemUtils::abort("Error: session end message insufficient.");
    }
    double totalReward = atof(s.c_str());

    delete sessionEndResponse;

    planner->finishSession(totalReward);
}

void IPCClient::initRound(vector<double>& initialState,
                          double& immediateReward) {
    stringstream os;
    os.str("");
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
       << "<round-request> <execute-policy>yes</execute-policy> "
          "</round-request>"
       << '\0';

    if (write(socket, os.str().c_str(), os.str().length()) == -1) {
        SystemUtils::abort("Error: writing to socket failed.");
    }

    XMLNode const* serverResponse = XMLNode::readNode(socket);

    if (!serverResponse || serverResponse->getName() != "round-init") {
        SystemUtils::abort("Error: round-request response insufficient.");
    }

    string s;
    if (!serverResponse->dissect("time-left", s)) {
        SystemUtils::abort("Error: round-request response insufficient.");
    }
    remainingTime = atoi(s.c_str());

    delete serverResponse;

    serverResponse = XMLNode::readNode(socket);

    readState(serverResponse, initialState, immediateReward);

    assert(MathUtils::doubleIsEqual(immediateReward, 0.0));

    delete serverResponse;

    planner->initRound(remainingTime);
}

void IPCClient::finishRound(XMLNode const* node, double& immediateReward) {
    // TODO: Move immediate rewards
    string s;
    if (!node->dissect("immediate-reward", s)) {
        SystemUtils::abort("Error: round end message insufficient.");
    }
    immediateReward = atof(s.c_str());

    if (!node->dissect("round-reward", s)) {
        SystemUtils::abort("Error: server communication failed.");
    }

    double roundReward = atof(s.c_str());

    planner->finishStep(immediateReward);
    planner->finishRound(roundReward);
}

/******************************************************************************
                         Submission of actions
******************************************************************************/

bool IPCClient::submitAction(vector<string>& actions, vector<double>& nextState,
                             double& immediateReward) {
    stringstream os;
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
       << "<actions>";

    for (unsigned int i = 0; i < actions.size(); ++i) {
        size_t cutPos = actions[i].find("(");
        if (cutPos == string::npos) {
            os << "<action><action-name>" << actions[i] << "</action-name>";
        } else {
            string actionName = actions[i].substr(0, cutPos);
            os << "<action><action-name>" << actionName << "</action-name>";
            string allParams = actions[i].substr(cutPos + 1);
            assert(allParams[allParams.length() - 1] == ')');
            allParams = allParams.substr(0, allParams.length() - 1);
            vector<string> params;
            StringUtils::split(allParams, params, ",");
            for (unsigned int j = 0; j < params.size(); ++j) {
                StringUtils::trim(params[j]);
                os << "<action-arg>" << params[j] << "</action-arg>";
            }
        }
        os << "<action-value>true</action-value></action>";
    }
    os << "</actions>" << '\0';
    if (write(socket, os.str().c_str(), os.str().length()) == -1) {
        return false;
    }
    XMLNode const* serverResponse = XMLNode::readNode(socket);

    bool roundContinues = true;
    if (serverResponse->getName() == "round-end") {
        finishRound(serverResponse, immediateReward);
        roundContinues = false;
    } else {
        readState(serverResponse, nextState, immediateReward);
        // to do
    }

    delete serverResponse;
    return roundContinues;
}

/******************************************************************************
                             Receiving of states
******************************************************************************/

void IPCClient::readState(XMLNode const* node, vector<double>& nextState,
                          double& immediateReward) {
    assert(node);
    assert(node->getName() == "turn");

    if (node->size() == 2 &&
        node->getChild(1)->getName() == "no-observed-fluents") {
        assert(false);
    }

    map<string, string> newValues;

    string s;
    if (!node->dissect("time-left", s)) {
        SystemUtils::abort("Error: turn response message insufficient.");
    }
    remainingTime = atoi(s.c_str());

    if (!node->dissect("immediate-reward", s)) {
        SystemUtils::abort("Error: turn response message insufficient.");
    }
    immediateReward = atof(s.c_str());

    for (int i = 0; i < node->size(); i++) {
        XMLNode const* child = node->getChild(i);
        if (child->getName() == "observed-fluent") {
            // 4550 todo

            readVariable(child, newValues);
        }
    }

    for (map<string, string>::iterator it = newValues.begin();
         it != newValues.end(); ++it) {
        string varName = it->first;
        string value = it->second;

        // If the variable has no parameters, its name is different from the one
        // that is used by PROST internally where no parents are used (afaik,
        // this changed at some point in rddlsim, and I am not sure if it will
        // change back which is why this hacky solution is fine for the moment).
        if (varName[varName.length() - 2] == '(') {
            varName = varName.substr(0, varName.length() - 2);
        }

        if (stateVariableIndices.find(varName) != stateVariableIndices.end()) {
            if (stateVariableValues[stateVariableIndices[varName]].empty()) {
                // TODO: This should be a numerical variable without
                // value->index mapping, but it can also be a boolean one atm.
                if (value == "true") {
                    nextState[stateVariableIndices[varName]] = 1.0;
                } else if (value == "false") {
                    nextState[stateVariableIndices[varName]] = 0.0;
                } else {
                    nextState[stateVariableIndices[varName]] =
                        atof(value.c_str());
                }
            } else {
                for (unsigned int i = 0;
                     i <
                     stateVariableValues[stateVariableIndices[varName]].size();
                     ++i) {
                    if (stateVariableValues[stateVariableIndices[varName]][i] ==
                        value) {
                        nextState[stateVariableIndices[varName]] = i;
                        break;
                    }
                }
            }
        }
    }
}

void IPCClient::readVariable(XMLNode const* node, map<string, string>& result) {
    string name;
    if (node->getName() != "observed-fluent") {
        assert(false);
    }

    if (!node->dissect("fluent-name", name)) {
        assert(false);
    }
    name = name.substr(0, name.length() - 1);

    vector<string> params;
    string value;
    string fluentName;

    for (int i = 0; i < node->size(); i++) {
        XMLNode const* paramNode = node->getChild(i);
        if (!paramNode) {
            assert(false);
            continue;
        } else if (paramNode->getName() == "fluent-arg") {
            string param = paramNode->getText();
            params.push_back(param.substr(0, param.length() - 1));
        } else if (paramNode->getName() == "fluent-value") {
            value = paramNode->getText();
            value = value.substr(0, value.length() - 1);
        }
    }
    name += "(";
    for (unsigned int i = 0; i < params.size(); ++i) {
        name += params[i];
        if (i != params.size() - 1) {
            name += ", ";
        }
    }
    name += ")";
    assert(result.find(name) == result.end());
    result[name] = value;
}

/******************************************************************************
                             Parser Interaction
******************************************************************************/

void IPCClient::executeParser(string const& taskDesc,
                              std::vector<ActionFluent*>* action_f_z) {
#ifdef NDEBUG
    std::string parserExec = "./rddl-parser-release ";
#else
    std::string parserExec = "./rddl-parser-debug ";
#endif
    Logger::logLine("Running RDDL parser at " + parserExec, Verbosity::VERBOSE);
    // Generate temporary input file for parser
    std::ofstream taskFile;
    stringstream taskFileNameStream;
    taskFileNameStream << "./parser_in_" << ::getpid() << ".rddl";
    string taskFileName = taskFileNameStream.str();
    taskFile.open(taskFileName.c_str());

    taskFile << taskDesc << endl;
    taskFile.close();

    stringstream parserOutStream;
    parserOutStream << "parser_out_" << ::getpid();
    string parserOut = parserOutStream.str();
    stringstream callString;
    callString << parserExec << taskFileName << " ./" << parserOut << " "
               << parserOptions;
    int result = std::system(callString.str().c_str());
    if (result != 0) {
        SystemUtils::abort("Error: " + parserExec + " had an error");
    }

    Parser parser(parserOut, action_f_z);
    parser.parseTask(stateVariableIndices, stateVariableValues,
                     &(this->action_indices));

    // 4550
    // 4550
    // 4550
    // 4550
    // 4550

    std::ifstream taskFile2;
    taskFile2.open("non-fluents-file");
    if (taskFile2.is_open()) {
        std::string line;
        while (getline(taskFile2, line)) {
            std::string name_ = erasechar(line);
            getline(taskFile2, line);
            std::string value_ = line;
            double value2 = std::stod(value_);
            (this->NonfluentsToValues)[name_] = value2;
            // std::cout<<
            // std::to_string((this->NonfluentsToValues)[name_])<<std::endl;
            // std::cout<< name_<<std::endl;
        }
        taskFile2.close();
    }

    std::ifstream taskFile22;
    taskFile22.open("CDFs_file_");
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
                this->sf2VariedSize_sfs[current_sf_name_] = tmp_quantifier_;
                tmp_quantifier_.clear();
                getline(taskFile22, line);
                while (line == "obj:") {
                    std::vector<std::string> obj_and_type;
                    getline(taskFile22, line);
                    obj_and_type.push_back(erasechar(line));
                    getline(taskFile22, line);
                    obj_and_type.push_back(erasechar(line));
                    getline(taskFile22, line);
                    if (this->sf_parameters_type_z.count(current_sf_name_) ==
                        0) {
                        std::set<std::vector<std::string>> tmp_pair_z;
                        tmp_pair_z.insert(obj_and_type);
                        sf_parameters_type_z[current_sf_name_] = tmp_pair_z;
                    } else {
                        sf_parameters_type_z[current_sf_name_].insert(
                            obj_and_type);
                    }
                }
            } else if (line == "fix-fluent_name:") {
                getline(taskFile22, line);
                if (this->sf2FixedSize_sfs.count(erasechar(current_sf_name_)) ==
                    0) {
                    std::set<std::string> tmp_sfs;
                    tmp_sfs.insert(erasechar(line));
                    sf2FixedSize_sfs[current_sf_name_] = tmp_sfs;
                } else {
                    this->sf2FixedSize_sfs[current_sf_name_].insert(
                        erasechar(line));
                }
            }
        }
        taskFile22.close();
    }
   if (true) {
        remove("non-fluents-file");
        remove("CDFs_file_");
    }

    std::ifstream taskFile23;
    taskFile23.open("objects_file_z");
    if (taskFile23.is_open()) {
        std::string line;
        while (getline(taskFile23, line)) {
            if (line == "object-and-type:") {
                getline(taskFile23, line);
                std::string obj_name_z = erasechar(line);
                getline(taskFile23, line);
                if (this->type_2_objs.count(erasechar(line)) == 0) {
                    std::set<std::string> tmp_;
                    tmp_.insert(obj_name_z);
                    type_2_objs[erasechar(line)] = tmp_;
                } else {
                    this->type_2_objs[erasechar(line)].insert(obj_name_z);
                }
            }
        }
        taskFile23.close();
    }

    remove("objects_file_z");

    // 4550

    // Remove temporary files

    if ((remove(taskFileName.c_str()) != 0) ||
        (remove(parserOut.c_str()) != 0)) {
        SystemUtils::abort("Error: deleting temporary file failed");
    }
}

std::string IPCClient::erasechar(std::string str) {
    std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
    str.erase(end_pos, str.end());
    return str;
}