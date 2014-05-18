#ifndef DATA_SPLITTER_H
#define	DATA_SPLITTER_H


#include <vector>
using namespace std;


#include<liblearning/core/dataset.h>
#include<liblearning/core/dataset_group.h>

namespace core
{

	class dataset_splitter
	{
	public:
		dataset_splitter(void);
		~dataset_splitter(void);

		virtual vector<vector<int>> split_impl(const dataset& data) const  = 0;

		dataset_group split(const dataset & data) const ;
	};


	class ordered_dataset_splitter :public dataset_splitter
	{
		int batch_num;
	public:
		ordered_dataset_splitter(int batch_num_);
		virtual vector<vector<int>> split_impl(const dataset& data) const;
	};

	class random_shuffer_dataset_splitter : public dataset_splitter
	{
		int batch_num;
	public:
		random_shuffer_dataset_splitter(int batch_num_);
		virtual vector<vector<int>> split_impl(const dataset& data) const;
	};

	class supervised_random_shuffer_dataset_splitter : public dataset_splitter
	{
		int batch_num;
	public:
		supervised_random_shuffer_dataset_splitter(int batch_num_);
		virtual vector<vector<int>> split_impl(const dataset& data) const ;
	};

	class random_shuffer_ratio_splitter: public dataset_splitter
	{
		const vector<NumericType> ratio;
	public:
		random_shuffer_ratio_splitter(const vector<NumericType> & ratio);
		virtual vector<vector<int>> split_impl(const dataset& data) const ;
	};


	class supervised_random_shuffer_ratio_splitter: public dataset_splitter
	{
		const vector<NumericType> ratio;
	public:
		supervised_random_shuffer_ratio_splitter(const vector<NumericType> & ratio);
		virtual vector<vector<int>> split_impl(const dataset& data) const ;
	};
}
#endif

