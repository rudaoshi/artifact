#include <vector>
#include <string>
#include <iostream>
using namespace std;

#define CATCH_CONFIG_MAIN
#include <artifact/test/catch.h>


#include <artifact/utils/batch_iterator.h>
#include <artifact/utils/matrix_io_txt.h>
#include <artifact/utils/matrix_utils.h>

using namespace artifact::utils;


SCENARIO( "bock iterator should work as expected", "[utils]" ) {

    GIVEN( "data is created" ) {

        const int sample_num = 10001;
        MatrixType X = MatrixType::Random(sample_num, 25);

        const int block_size = 5000;

        WHEN("block iterator is move forward one step")
        {
            batch_iterator< MatrixType> iter(&X, block_size);

            iter ++;
            THEN(" it points to the correct region ")
            {
                REQUIRE(*iter == X.block(block_size, 0, block_size, 25));

            }
        }

        WHEN("block iterator is move to the end of the region")
        {
            batch_iterator< MatrixType> iter(&X, block_size);

            iter ++;
            iter ++;
            THEN(" it points to the correct region ")
            {
                REQUIRE(*iter == X.block(block_size*2, 0, 1, 25));
            }
        }

        WHEN("block iterator is move out of the region")
        {
            batch_iterator< MatrixType> iter(&X, block_size);

            iter ++;
            iter ++;
            iter ++;
            THEN(" it should be judged false ")
            {
                REQUIRE(not iter);
            }
        }



    };
}

