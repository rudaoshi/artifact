/*
 * network_optimize_objective.h
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#ifndef NETWORK_OPTIMIZE_OBJECTIVE_H_
#define NETWORK_OPTIMIZE_OBJECTIVE_H_

#include <liblearning/optimization/optimize_objective.h>
#include "deep_auto_encoder.h"
#include "network_objective.h"


namespace deep
{
	class network_optimobj_adapter: public optimization::optimize_objective
	{

		deep_auto_encoder & net;
		network_objective & obj;


	public:
		network_optimobj_adapter(	deep_auto_encoder & net, network_objective & obj);
		virtual ~network_optimobj_adapter();

		virtual NumericType value(const VectorType & x);
		virtual tuple<NumericType, VectorType> value_diff(const VectorType & x);

		virtual void progress_notification(const VectorType & x, int finetuneIterNum);


	};
}
#endif /* NETWORK_OPTIMIZE_OBJECTIVE_H_ */
