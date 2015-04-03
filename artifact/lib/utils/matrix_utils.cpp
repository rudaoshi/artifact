
#include <algorithm>
#include <tuple>
using namespace std;

#include <artifact/config.h>


tuple<MatrixType, VectorType> shuffle_data_set(const MatrixType & X,
    const VectorType & y)
{

    VectorXi indices = VectorXi::LinSpaced(X.cols(), 0, X.cols());
    std::random_shuffle(indices.data(), indices.data() + X.cols());
    //the following statement is evaluated "in-place", without any temporary. So this is definitely the right way to go.
    MatrixType shuffled_X = X * indices.asPermutation();
    VectorType shuffled_y = y * indices.asPermutation();

    return make_tuple(shuffled_X, shuffled_y);
}
