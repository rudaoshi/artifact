/*
 * network_optimize_objective.cpp
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#include <liblearning/deep/network_optimobj_adapter.h>


namespace deep
{
	network_optimobj_adapter::network_optimobj_adapter(	deep_auto_encoder & net_, network_objective & obj_)
	:net(net_),obj(obj_)
	{
		// TODO Auto-generated constructor stub

	}

	network_optimobj_adapter::~network_optimobj_adapter()
	{
		// TODO Auto-generated destructor stub
	}


	NumericType network_optimobj_adapter::value(const VectorType & x)
	{
		net.set_Wb(x);
		return obj.value(net);
	}


	tuple<NumericType, VectorType> network_optimobj_adapter::value_diff(const VectorType & x)
	{

		net.set_Wb(x);

		return obj.value_diff(net);

	}

	void network_optimobj_adapter::progress_notification(const VectorType & x,int iterNum)
	{
		net.set_Wb(x);

		net.progress_notified( iterNum);

//		std::cout << obj.get_info() << std::endl;
	}
}
