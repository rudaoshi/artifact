#include <vector>
#include <string>
#include <iostream>
using namespace std;

#include <boost/timer/timer.hpp>

using boost::timer::cpu_timer;
using boost::timer::cpu_times;
using boost::timer::nanosecond_type;



#define CATCH_CONFIG_MAIN
#include <artifact/test/catch.h>


#include <artifact/network/deep_network.h>
#include <artifact/network/network_creator.h>
#include <artifact/network/network_trainer.h>
#include <artifact/utils/matrix_io_txt.h>
#include <artifact/optimization/numerical_gradient.h>
#include <artifact/optimization/mt_sgd_optimizer.h>

using namespace artifact::network;
using namespace artifact::utils;
using namespace artifact::optimization;

SCENARIO( "dnn can be created and operated correctly", "[dnn_prediction]" ) {

    GIVEN( "network created" ) {

        vector<int> layer_sizes = {25,500,500,1000,500,500,1};
        vector<string> layer_types = {"logistic","logistic","logistic","logistic","logistic","logistic"};
        string net_loss = "mse";

        network_architecture arch;
        arch.layer_sizes = layer_sizes;
        arch.activator_types = layer_types;
        arch.loss = net_loss;

        random_network_creator creator;
        deep_network net = creator.create(arch);

        const int sample_num = 10000;
        MatrixType X = MatrixType::Random(sample_num, 25);
        MatrixType y = MatrixType::Random(sample_num, 1) ;

        WHEN("network can be trained efficiently with single thread")
        {
            mt_sgd_optimizer optimizer_;
            optimizer_.learning_rate = 0.001;
            optimizer_.decay_rate = 0.9;
            optimizer_.max_epoches = 5;
            optimizer_.batch_per_thread = 2000;
            optimizer_.thread_num = 4;


            nanosecond_type const one_min(60 * 1000000000LL);
            nanosecond_type const half_min(30 * 1000000000LL);
            nanosecond_type const quater_min(15 * 1000000000LL);
            optimization_trainer trainer(optimizer_);

            cpu_timer timer;
            net = trainer.train(net,X,&y);
            cpu_times const elapsed_times(timer.elapsed());
            nanosecond_type const elapsed(elapsed_times.wall);


            THEN(" it should be finished in 15s ")
            {
                std::cout << "Runing Time: " << elapsed << std::endl;
                if (typeid(NumericType) == typeid(double)) {
                    REQUIRE(elapsed <= half_min);
                }
                else if  (typeid(NumericType) == typeid(float))
                {
                    REQUIRE(elapsed <= quater_min);
                }
            }
        }



    };
}
