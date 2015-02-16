
#import <artifact/loss/loss.h>


using namespace artifact::losses;

NumericType mse_loss::loss(const MatrixType &x,
            const VectorType & y) {

    if (x.rows() != 1)
    {
        throw runtime_error("MSE Loss accecpt vector outputs, not a matrix");
    }

    return (x.row(0) - y.transpose()).array().pow(2).sum();

}

MatrixType mse_loss::gradient(const MatrixType &x,
            const VectorType & y)
{
    return 2*(x.row(0) - y.transpose());
}