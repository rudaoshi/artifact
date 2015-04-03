#include <artifact/optimization/gd_optimizer.h>

using namespace artifact::optimization;

gd_optimizer::gd_optimizer()
{
    this->learning_rate = 0.01;
    this->decay_rate = 0.9;
    this->max_epoches = 10;
}


gd_optimizer::~gd_optimizer(void)
{
}


VectorType gd_optimizer::optimize(optimizable & obj,
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
        opt.set_parameter(param);
		tie(fval,grad) = opt.gradient(X, y);

        param -= cur_learning_rate*grad/grad.norm();
        cur_learning_rate *= this->decay_rate;
	}

	return param;
}

