#include <liblearning/core/dataset_group.h>

#include <liblearning/core/dataset.h>
namespace core
{
dataset_group::dataset_group(void)
{
}


dataset_group::~dataset_group(void)
{
}

void dataset_group::add_dataset(const shared_ptr<dataset> & data)
{
	datasets.push_back( data);
}

shared_ptr<dataset> dataset_group::get_dataset(int i)
{
	return  datasets[i];
}

int dataset_group::get_dataset_num()
{
	return datasets.size();
}
}