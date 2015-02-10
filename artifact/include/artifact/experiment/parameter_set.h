#ifndef PARAMETER_SET_H
#define PARAMETER_SET_H
#include <liblearning/core/config.h>
#include <vector>
using namespace std;

namespace experiment
{
class parameter_set
{
	vector<vector<NumericType>> param_candidates;

public:
	parameter_set(void);
	~parameter_set(void);

	void add_param_candidate(const vector<NumericType> & candidate);

	vector<vector<NumericType>> emurate_parameter_combination();

	int get_param_num();

	const vector<NumericType> & get_param_candidate(int i);
};
}
#endif

