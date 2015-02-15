#ifndef ARTIFACT_MACHINE_H_
#define ARTIFACT_MACHINE_H_

#include <artifact/config.h>


namespace artifact
{
    namespace machines {
        /* machine_maker class.
         *
         * machine_maker builds machines from input data. All training algorithms
         * in this library are subclass of machine_maker
         */

        class machine {

        public:

            virtual VectorType predict(const VectorType &testdata) = 0;

            virtual MatrixType
                    predict(const MatrixType &test_set) = 0;
        };





    }


}
#endif
