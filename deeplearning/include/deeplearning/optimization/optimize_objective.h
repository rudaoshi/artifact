/*
 * optimize_objective.h
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#ifndef OPTIMIZE_OBJECTIVE_H_
#define OPTIMIZE_OBJECTIVE_H_

#include <liblearning/core/config.h>

#include <tuple>
using namespace std;

namespace optimization
{
	class optimize_objective
	{
	public:
		optimize_objective();
		virtual ~optimize_objective();

		virtual NumericType value(const VectorType & x) = 0;
		virtual tuple<NumericType, VectorType> value_diff(const VectorType & x) = 0;

		virtual void progress_notification(const VectorType & x, int iter) = 0;
	};
}

#endif /* OPTIMIZE_OBJECTIVE_H_ */
