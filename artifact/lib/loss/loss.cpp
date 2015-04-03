
#include <artifact/loss/loss.h>


using namespace artifact::losses;

NumericType mse_loss::loss(const MatrixType &x,
            const MatrixType * y) {


    return (x - (*y)).array().pow(2).sum();

}

MatrixType mse_loss::gradient(const MatrixType &x,
            const MatrixType * y)
{
    return 2*(x - (*y));
}