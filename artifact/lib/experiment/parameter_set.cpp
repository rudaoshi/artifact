#include <liblearning/experiment/parameter_set.h>
using namespace experiment;

parameter_set::parameter_set(void)
{
}


parameter_set::~parameter_set(void)
{
}



void parameter_set::add_param_candidate(const vector<NumericType> & candidate)
{
	param_candidates.push_back(candidate);
}

vector<vector<NumericType>>  generate_param_combination(const vector<vector<NumericType>> & candidates)
{
	if (candidates.size() == 0)
	{
		vector<vector<NumericType>> param_comb;
		return param_comb;
	}

	if (candidates.size() == 1)
	{
		vector<vector<NumericType>> param_comb(candidates[0].size());
		for (int i = 0;i<candidates[0].size();i++)
		{
			param_comb[i].push_back(candidates[0][i]);
		}

		return param_comb;
	}

	vector<vector<NumericType>> temp_cand = candidates;
	temp_cand.erase(temp_cand.begin());

	vector<vector<NumericType>> temp_comb =  generate_param_combination(temp_cand);

	vector<vector<NumericType>> comb(candidates[0].size()*temp_comb.size());

	for (int i = 0;i  < candidates[0].size();i++)
	{
		for (int j = 0; j < temp_comb.size();j++)
		{
			comb[i*temp_comb.size()+j].push_back(candidates[0][i]); 
			comb[i*temp_comb.size()+j].insert(comb[i*temp_comb.size()+j].end(),temp_comb[j].begin(),temp_comb[j].end());
		}
	}

	return comb;
}


vector<vector<NumericType>> parameter_set::emurate_parameter_combination()
{
	return generate_param_combination( param_candidates);
}

int parameter_set::get_param_num()
{
	return param_candidates.size();
}

const vector<NumericType> & parameter_set::get_param_candidate(int i)
{
	return param_candidates[i];
}