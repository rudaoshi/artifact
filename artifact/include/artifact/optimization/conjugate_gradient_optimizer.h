/*
 * conjugate_gradient_optimizer.h
 *
 *  Created on: 2010-6-20
 *      Author: sun
 */

#ifndef CONJUGATE_GRADIENT_OPTIMIZER_H_
#define CONJUGATE_GRADIENT_OPTIMIZER_H_
#include <liblearning/core/config.h>
#include "optimizer.h"

namespace optimization
{
	class conjugate_gradient_optimizer: public optimizer
	{
		int max_iter;

		NumericType ftol;

		int iter;

	public:
		conjugate_gradient_optimizer(int max_iter_, NumericType ftol);
		virtual ~conjugate_gradient_optimizer();


		virtual tuple<NumericType, VectorType> optimize(optimize_objective& obj, const VectorType & p);

		virtual shared_ptr<optimizer> clone();
	};
}

#endif /* CONJUGATE_GRADIENT_OPTIMIZER_H_ */
