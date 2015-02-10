#ifndef DATASET_GROUP
#define DATASET_GROUP

#include "config.h"
#include <vector>
#include <string>

using namespace std;


namespace core
{

	class dataset;

	class dataset_group
	{

		vector<shared_ptr<dataset>> datasets;

	public:
		dataset_group(void);
		~dataset_group(void);

		void add_dataset(const shared_ptr<dataset> & data);

		shared_ptr<dataset>  get_dataset(int i);

		int get_dataset_num();
	};
}
#endif