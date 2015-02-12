#include <artifact/optimization/gradient_descent_optimizer.h>

using namespace optimization;

gd_optimizer::gd_optimizer(
        NumericType learning_rate_, NumericType decay_rate_, int max_epoches_)
	: learning_rate(learning_rate_),decay_rate(decay_rate_),max_epoches(max_epoches_)
{
}


gd_optimizer::~gd_optimizer(void)
{
}


tuple<NumericType, VectorType> gd_optimizer::optimize(optimizable & obj,
        const VectorType & param0,
        const MatrixType & X,
        const VectorType * y // nullptr for unsupervised optimizer
)
{
	NumericType fval = numeric_limits<NumericType>::quiet_NaN();

	VectorType diff;

	VectorType param = param0;

    NumericType cur_learning_rate = this->learning_rate;
	for (int i = 0; i < this->max_epoches; i++)
	{
        obj.set_parameter(param);
		tie(fval,diff) = obj.gradient(X, *y);

        param -= cur_learning_rate*diff;
        cur_learning_rate *= this->decay_rate;
	}

    obj.set_parameter(param);
    fval = obj.objective(X, *y);

	return make_tuple(fval,param);
}

