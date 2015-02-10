#ifndef EXPERIMENT2_H_
#define EXPERIMENT2_H_

#include "parameter_set.h"
#include "experiment_datasets.h"

#include <algorithm>
#include <string>
#include <fstream>
#include <tuple>
using namespace std;

#include <boost/filesystem.hpp> 
#include <boost/thread.hpp>
#include <boost/thread/thread.hpp>

#include <boost/multi_array.hpp>


namespace experiment
{
	template<typename M>
	class experiment2
	{

	protected:

		string experiment_name;

		const experiment_datasets & datasets;
	
		parameter_set   params;

		ofstream logfile;

	public:
		experiment2(const experiment_datasets & datasets_, const string & experiment_name_):datasets(datasets_), experiment_name(experiment_name_)
		{
			//if ( boost::filesystem::exists( experiment_name ))
			//{
			//	throw "the log file with same name :" + experiment_name + " has already exist!";
			//}

			logfile.open((experiment_name+".log").c_str(),ios_base::app);
		}
		~experiment2(void)
		{
			logfile.close();
		}

	public:

		virtual tuple<shared_ptr<dataset>, shared_ptr<dataset>> prepare_dataset(const shared_ptr<dataset> & train, const shared_ptr<dataset> & test) = 0 ;

		virtual shared_ptr<M> train_one_machine(const shared_ptr<dataset> & train, const vector<NumericType> & params) = 0 ;

		//virtual NumericType test_performance(M & , const dataset & test) = 0 ;

		//virtual bool save_machine(const M & machine, const string & file_name) = 0;

	public:


		void set_param_candidate(const parameter_set & params_)
		{
			params = params_;
		}


	protected:

		void log_experiment_configuration()
		{
			logfile << " The configuration of experiment : " << endl;
			logfile << " Training-testing pairs : " << datasets.get_train_test_pair_num() << endl;
			logfile << " Validation folder nums for each training set : " ;
			for (int i = 0;i < datasets.get_train_test_pair_num();i++)
			{
				logfile << datasets.get_train_test_pair(i).get_tv_folder_num() << " " ;
			}
			logfile << endl << endl;

			logfile << " There are " << params.get_param_num() <<  " parameter candidates :" << endl ;
			for (int i = 0; i < params.get_param_num(); i ++ )
			{
				logfile << "\t the candidates for " << i <<  " -th learning parameter are :" ;
				const vector<NumericType> & cur_candidate = params.get_param_candidate(i);
				for_each(cur_candidate.begin(),cur_candidate.end(),[this](NumericType param){logfile << param << " " ;});
				logfile << endl;
			}
			logfile << endl << endl;
		}

	

	public:

		virtual NumericType evaluate() = 0;
	
	};
}

#endif
