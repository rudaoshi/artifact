#include <liblearning/optimization/step_update_optimizer.h>

using namespace optimization;

step_update_optimizer::step_update_optimizer(NumericType step_size_, int iter_num_)
	: step_size(step_size_),iter_num(iter_num_)
{
}


step_update_optimizer::~step_update_optimizer(void)
{
}


tuple<NumericType, VectorType> step_update_optimizer::optimize(optimize_objective& obj, const VectorType & p)
{
	NumericType fval = numeric_limits<NumericType>::quiet_NaN();

	VectorType diff(p.size());

	VectorType result = p;

	for (int i = 0;i < iter_num;i++)
	{
		tie(fval,diff) = obj.value_diff(result);

		result -= step_size*diff;

		obj.progress_notification(result, i);
	}

	return make_tuple(fval,result);
}

shared_ptr<optimizer> step_update_optimizer::clone()
{
	shared_ptr<optimizer> ptr(new step_update_optimizer(*this));

	return ptr;

}
