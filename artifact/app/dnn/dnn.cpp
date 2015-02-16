

#include <string>
#include <vector>
#include <iostream>
using namespace std;

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

#include <artifact/network/deep_network.h>
#include <artifact/network/network_creator.h>
#include <artifact/network/network_trainer.h>
#include <artifact/utils/matrix_io_txt.h>

using namespace artifact::network;
using namespace artifact::utils;



int main(int argc, char ** argv)
{

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("config_file,c", po::value<string>(), "path to config file")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    if (not vm.count("config_file")) {
        cerr << "Config file path is not specificed.";
        return -1;
    }

    string config_file_path = vm["config_file"].as<string>();

    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(config_file_path, pt);
    string train_data_path_prefix = pt.get<std::string>("data.train_prefix");
//    string valid_data_path = pt.get<std::string>("data.valid");

    string net_layer_sizes = pt.get<std::string>("network.layer_size");
    string net_layer_types = pt.get<std::string>("network.layer_types");
    string net_loss = pt.get<std::string>("network.loss");


    int max_epoches = pt.get<int>("train.max_epoches");
    float learning_rate = pt.get<float>("train.learning_rate");
    float decay_rate = pt.get<float>("train.decay_rate");

    vector< string > layer_size_str_vector; // #2: Search for tokens
    boost::split( layer_size_str_vector, net_layer_sizes, boost::is_any_of(",") );
    vector<int> layer_sizes;
    for (string size_str: layer_size_str_vector)
    {
        layer_sizes.push_back(boost::lexical_cast<int>(size_str));
    }

    vector< string > layer_type_vector;
    boost::split( layer_type_vector, net_layer_types, boost::is_any_of(",") );

    network_architecture arch;
    arch.layer_sizes = layer_sizes;
    arch.activator_types = layer_type_vector;
    arch.loss = net_loss;

    random_network_creator creator;
    deep_network net = creator.create(arch);

    gd_training_setting training_settings;
    training_settings.learning_rate = learning_rate;
    training_settings.decay_rate = decay_rate;
    training_settings.max_epoches = max_epoches;

    gd_network_trainer trainer;

    MatrixType X = load_matrix_from_txt(train_data_path_prefix + ".X");
    VectorType y = load_vector_from_txt(train_data_path_prefix + ".y");


    net = trainer.train(net,training_settings,X,&y);

}