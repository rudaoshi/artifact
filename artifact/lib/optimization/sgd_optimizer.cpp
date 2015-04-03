#include <artifact/optimization/sgd_optimizer.h>
#include <artifact/utils/batch_iterator.h>

using namespace artifact::optimization;
using namespace artifact::utils;

sgd_optimizer::sgd_optimizer()
{
    this->max_epoches = 10;
    this->learning_rate = 0.01;
    this->decay_rate = 0.9;
    this->batch_size = 10000;
}


sgd_optimizer::~sgd_optimizer(void)
{
}


VectorType sgd_optimizer::optimize(optimizable & obj,
        const VectorType & param0,
        const MatrixType & X,
        const MatrixType * y // nullptr for unsupervised optimizer
)
{
    gradient_optimizable & opt = dynamic_cast<gradient_optimizable &>(obj);

	NumericType fval = numeric_limits<NumericType>::quiet_NaN();

	VectorType grad;

	VectorType param = param0;

    NumericType cur_learning_rate = this->learning_rate;
	for (int i = 0; i < this->max_epoches; i++)
	{

        batch_iterator<MatrixType> X_iter(&X, this->batch_size);
        batch_iterator<MatrixType> y_iter(y, this->batch_size);
        for (; X_iter; ++X_iter, ++ y_iter ) {
            opt.set_parameter(param);
            MatrixType cur_X = *X_iter;

            if (y_iter)
            {
                MatrixType cur_y = *y_iter;
                tie(fval,grad) = opt.gradient(cur_X, &cur_y);

            }
            else {
                tie(fval, grad) = opt.gradient(cur_X, 0);

            }

            param -= cur_learning_rate*grad/grad.norm();
        }

        cur_learning_rate *= this->decay_rate;
	}

	return param;
}

