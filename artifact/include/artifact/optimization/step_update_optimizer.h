#ifndef STEP_UPDATE_OPTIMIZER_H
#define STEP_UPDATE_OPTIMIZER_H



#include <liblearning/optimization/optimizer.h>



namespace optimization
{

	class step_update_optimizer :
		public optimizer
	{
		NumericType step_size;

		int iter_num;

	public:
		step_update_optimizer(NumericType step_size, int iter_num);
		virtual ~step_update_optimizer(void);


		virtual tuple<NumericType, VectorType> optimize(optimize_objective& obj, const VectorType & p);

		virtual shared_ptr<optimizer> clone();
	};

}
#endif
