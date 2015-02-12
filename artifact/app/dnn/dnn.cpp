
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

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


    string epoches = pt.get<std::string>("train.epoches");
    string learning_rate = pt.get<std::string>("train.learning_rate");
    string decay_rate = pt.get<std::string>("train.decay_rate");
}