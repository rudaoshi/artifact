
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
using namespace boost;

#include <artifact/network/deep_network.h>
#include <artifact/network/network_creater.h>
#include <artifact/network/network_trainer.h>

int main(int argc, char ** argv)
{

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("config_file", po::value<string>(), "path to config file")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    if (not vm.count("config_file")) {
        cout << "Config file path is not specificed.";
        return -1;
    }

    string config_file_path = vm["config_file"];

    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(config_file_path, pt);
    string train_data_path = pt.get<std::string>("data.train");
    string valid_data_path = pt.get<std::string>("data.valid");

    string net_layer_sizes = pt.get<std::string>("network.layer_size");
    string net_layer_types = pt.get<std::string>("network.layer_types");
    string net_loss = pt.get<std::string>("network.loss");


    int max_epoches = pt.get<int>("train.max_epoches");
    float learning_rate = pt.get<float>("train.learning_rate");
    float decay_rate = pt.get<float>("train.decay_rate");

    vector< string > layer_size_str_vector; // #2: Search for tokens
    split( layer_size_str_vector, net_layer_sizes, is_any_of(",") );
    vector<int> layer_sizes;
    for (string size_str: layer_size_str_vector)
    {
        layer_sizes.push_back(lexical_cast<int>(layer));
    }

    vector< string > layer_type_vector; // #2: Search for tokens
    split( layer_type_vector, net_layer_types, is_any_of(",") );

    network_architecture arch;
    arch.layer_sizes = layer_sizes;
    arch.activator_types = layer_type_vector;
    arch.loss = net_loss;

    network_creator creator;
    deep_network net = creator.create(arch);

    gd_training_param training_param;
    training_param.learning_rate = lexical_cast<NumericType>(learning_rate_str);
    training_param.decay_rate = lexical_cast<NumericType>(decay_rate_str);
    training_param.max_epoches = lexical_cast<NumericType>(max_epoches_str);

    gd_network_trainer trainer;

    net = trainer.train(net,X,y,training_param);

}