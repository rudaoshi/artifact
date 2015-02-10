#ifndef MACHINE_H_
#define MACHINE_H_

#include <artifact/config.h>


namespace artifact
{
    namespace core {
        /* machine_maker class.
         *
         * machine_maker builds machines from input data. All training algorithms
         * in this library are subclass of machine_maker
         */
        template<typename Machine>
        class machine_maker {
        public:

            virtual Machine train(
                    const typename sample_set_type<Machine::sample_type>::type &train_data
            ) = 0;

        };

        class machine {

        public:

            virtual NumericType predict(const VectorType &testdata) = 0;

            virtual VectorType
                    predict(const MatrixType &test_set) = 0;
        };





    }


}
#endif
