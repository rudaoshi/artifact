
#ifndef ARTIFACT_NETWORK_NETWORK_CREATEOR_H_
#define ARTIFACT_NETWORK_NETWORK_CREATEOR_H_


#include <vector>
#include <string>
using namespace std;

#include <artifact/network/deep_network.h>


namespace artifact
{
    namespace network
    {
        struct network_architecture
        {
            vector<int> layer_sizes;
            vector<string> activator_types;
            string loss;

        };

        struct create_context
        {

        };
        class network_creator
        {
        public:
            virtual deep_network create(const network_architecture & architec_param,
                                const create_context * context = 0) = 0;

        };

        class random_network_creator: public network_creator
        {
        public:
            virtual deep_network create(const network_architecture & architec_param,
                    const create_context * context = 0);
        };
    }
}

#endif