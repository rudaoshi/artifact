#include <liblearning/core/data_splitter.h>

#include <liblearning/core/dataset.h>
#include <boost/foreach.hpp>
using namespace core;
dataset_splitter::dataset_splitter(void)
{
}


dataset_splitter::~dataset_splitter(void)
{
}

dataset_group dataset_splitter::split(const dataset & data) const
{
	dataset_group group;

	vector<vector<int>> batch_ids = this->split_impl(data);

	for (int i = 0;i<batch_ids.size();i++)
	{

		group.add_dataset(data.sub_set(batch_ids[i]));
	}

	return group;
}


ordered_dataset_splitter ::ordered_dataset_splitter(int batch_num_):batch_num(batch_num_)
{

}
				
vector<vector<int>> ordered_dataset_splitter ::split_impl(const dataset& data) const
{
	vector<vector<int>> batch_ids(batch_num);
	int sample_num = data.get_sample_num();
	int batch_size = ceil(float(sample_num)/batch_num);

	for (int i = 0;i<batch_num;i++)
	{
		int cur_batch_size = batch_size;
		if (i == batch_num-1)
			cur_batch_size = sample_num - (batch_num-1)*batch_size;
		vector<int> cur_batch_id(cur_batch_size);

		for (int j = 0;j<cur_batch_size;j++)
			cur_batch_id[j] = i*batch_size + j;

		batch_ids.push_back(cur_batch_id);

	}

	return batch_ids;
}



random_shuffer_dataset_splitter::random_shuffer_dataset_splitter(int batch_num_):batch_num(batch_num_)
{
}

#include <algorithm>
vector<vector<int>> random_shuffer_dataset_splitter ::split_impl(const dataset& data) const
{
	vector<vector<int>> batch_ids(batch_num);

	int sample_num = data.get_sample_num();
	vector<int> temp(sample_num);
	for (int i = 0;i<sample_num;i++)
		temp[i] = i;

	std::random_shuffle ( temp.begin(), temp.end() );


	int batch_size = ceil(float(sample_num)/batch_num);

	for (int i = 0;i<batch_num;i++)
	{
		int cur_batch_size = batch_size;
		if (i == batch_num-1)
			cur_batch_size = sample_num - (batch_num-1)*batch_size;
		vector<int> cur_batch_id(cur_batch_size);

		for (int j = 0;j<cur_batch_size;j++)
			cur_batch_id[j] = temp[i*batch_size + j];

		batch_ids[i] = cur_batch_id;

	}

	return batch_ids;
}
supervised_random_shuffer_dataset_splitter ::supervised_random_shuffer_dataset_splitter(int batch_num_):batch_num(batch_num_)
{

}

#include <liblearning/core/supervised_dataset.h>

vector<vector<int>> supervised_random_shuffer_dataset_splitter ::split_impl(const dataset& data) const
{
	const supervised_dataset & s_data_set = dynamic_cast<const supervised_dataset &>(data);

	const vector<int> & label = s_data_set.get_label();
	const vector<int> & class_id = s_data_set.get_class_id();
	vector<vector<int>> batches(batch_num);
	for (int i = 0;i<class_id.size();i++)
	{
		vector<int> cur_class_samples;

		for (int j = 0;j<label.size();j++)
		{
			if (label[j] == class_id[i]) cur_class_samples.push_back(j);
		}
		
		shared_ptr<dataset> cur_class_data_set = s_data_set.sub_set(cur_class_samples);

		random_shuffer_dataset_splitter rand_sh_maker(batch_num);
		vector<vector<int>> cur_class_batches = rand_sh_maker.split_impl(*cur_class_data_set);

		for (int j = 0;j<cur_class_batches.size();j++)
		{
			for (int k = 0; k <cur_class_batches[j].size();k++)
			{
				cur_class_batches[j][k] = cur_class_samples[cur_class_batches[j][k]];
			}
			batches[j].insert(batches[j].end(),cur_class_batches[j].begin(),cur_class_batches[j].end());
		}
	}

	return batches;
	
}

random_shuffer_ratio_splitter ::random_shuffer_ratio_splitter(const vector<NumericType> & ratio_):ratio(ratio_)
{

}

#include <boost/iterator/counting_iterator.hpp>
#include <numeric>
#include <algorithm>

vector<vector<int>> random_shuffer_ratio_splitter ::split_impl(const dataset& data) const
{
	vector<NumericType> percent(ratio);
	
	NumericType total = std::accumulate(ratio.begin(),ratio.end(),0);

	BOOST_FOREACH(NumericType & x,percent){ x = x/total; }
//	std::transform(percent.begin(),percent.end(),percent.begin(),[total](NumericType val){return val/total;});

	vector<vector<int>> group_ids(percent.size());

	int sample_num = data.get_sample_num();
	vector<int> temp;

	std::copy(
		boost::counting_iterator<unsigned int>(0),
		boost::counting_iterator<unsigned int>(sample_num), 
		std::back_inserter(temp));

	std::random_shuffle ( temp.begin(), temp.end() );

	vector<int>::iterator cur_begin_iter = temp.begin();
	for (int i = 0;i<percent.size();i++)
	{
		int cur_group_size = floor(sample_num * percent[i]);
		vector<int>::iterator  cur_end_iter = cur_begin_iter + cur_group_size;
		if (i == percent.size()-1)
		{
			cur_end_iter = temp.end();
			cur_group_size = cur_end_iter - cur_begin_iter;
		}
		vector<int> cur_group_id(cur_group_size);

		copy(cur_begin_iter,cur_end_iter, cur_group_id.begin());

		cur_begin_iter = cur_end_iter;

		group_ids[i] = cur_group_id;

	}

	return group_ids;
	
}


supervised_random_shuffer_ratio_splitter ::supervised_random_shuffer_ratio_splitter(const vector<NumericType> & ratio_):ratio(ratio_)
{

}


vector<vector<int>> supervised_random_shuffer_ratio_splitter ::split_impl(const dataset& data) const
{
	const supervised_dataset & s_data_set = dynamic_cast<const supervised_dataset &>(data);

	const vector<int> & label = s_data_set.get_label();
	const vector<int> & class_id = s_data_set.get_class_id();
	vector<vector<int>> groups(ratio.size());
	for (int i = 0;i<class_id.size();i++)
	{
		vector<int> cur_class_samples;

		for (int j = 0;j<label.size();j++)
		{
			if (label[j] == class_id[i]) cur_class_samples.push_back(j);
		}
		
		shared_ptr<dataset> cur_class_data_set = s_data_set.sub_set(cur_class_samples);

		random_shuffer_ratio_splitter rand_sh_maker(ratio);
		vector<vector<int>> cur_class_batches = rand_sh_maker.split_impl(*cur_class_data_set);

		for (int j = 0;j<cur_class_batches.size();j++)
		{
			for (int k = 0; k <cur_class_batches[j].size();k++)
			{
				cur_class_batches[j][k] = cur_class_samples[cur_class_batches[j][k]];
			}
			groups[j].insert(groups[j].end(),cur_class_batches[j].begin(),cur_class_batches[j].end());
		}
	}

	return groups;
	
}