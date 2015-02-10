#include <liblearning/deep/objective/self_related_network_objective.h>

using namespace deep;
using namespace deep::objective;

self_related_network_objective::self_related_network_objective(void)
{
	type = self_related;
}


self_related_network_objective::~self_related_network_objective(void)
{
}

void self_related_network_objective::set_dataset(const shared_ptr<dataset> & data_set)
{
}