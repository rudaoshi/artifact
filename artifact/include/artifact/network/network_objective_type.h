#ifndef NETWORK_OBJECTIVE_TYPE_H
#define NETWORK_OBJECTIVE_TYPE_H

namespace deep
{
	enum network_objective_type
	{
		self_related = 0,							// not related to the data sets
		encoder_related = 1,                        // related to the output of upto encoder layer
		decoder_related = 2                        // related to the output of upto decoder layer

	};
}


#endif
