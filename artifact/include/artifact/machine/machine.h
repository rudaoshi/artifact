#ifndef ARTIFACT_MACHINE_H_
#define ARTIFACT_MACHINE_H_

#include <artifact/config.h>


namespace artifact
{
    namespace machines {

        /**
        * machine class
        *
        * Machine predict something for a data set.
        * The data set is a matrix. Rows are samples and columns are observations/features.
        */
        class machine {

        public:

            virtual RowVectorType predict(const RowVectorType &testdata) = 0;

            virtual MatrixType
                    predict(const MatrixType & X) = 0;
        };





    }


}
#endif
