/*
 * optimizer.h
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <liblearning/core/config.h>
#include <tuple>

using namespace std;

#include "optimize_objective.h"

namespace optimization
{
	class optimizer
	{
	public:
		optimizer();
		virtual ~optimizer();


		virtual tuple<NumericType, VectorType> optimize(optimize_objective& obj, const VectorType & x0) = 0;

		virtual shared_ptr<optimizer> clone() = 0;
	};
}

#endif /* OPTIMIZER_H_ */
