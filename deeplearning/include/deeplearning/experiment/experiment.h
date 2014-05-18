#ifndef EXPERIMENT_H_
#define EXPERIMENT_H_

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
#include <boost/foreach.hpp>
#include <boost/algorithm/string/join.hpp>
#include <liblearning/core/platform.h>
namespace experiment
{
  using namespace boost::filesystem;
  using namespace core;


	template<typename M>
	class experiment
	{

	protected:


		string dataset_name;

		string remark;

		const experiment_datasets & datasets;
	
		parameter_set   train_candidate;
    
		parameter_set   test_candidate;

		log4cpp::Category * log;

		bool preprocess;

		
		
		

	public:
	experiment(const experiment_datasets & datasets_):datasets(datasets_)
		{
			//if ( boost::filesystem::exists( experiment_name ))
			//{
			//	throw "the log file with same name :" + experiment_name + " has already exist!";
			//}


		  preprocess = true;
		}
		~experiment(void)
		{

		}

	public:

		virtual tuple<shared_ptr<dataset>, shared_ptr<dataset>> prepare_dataset(const shared_ptr<dataset> & train, const shared_ptr<dataset> & test) = 0 ;

		virtual shared_ptr<M> train_one_machine(const string & machineID_,const shared_ptr<dataset> & train, const vector<NumericType> & train_params, const shared_ptr<dataset> & test) = 0 ;

		virtual NumericType test_performance(M & ,const shared_ptr<dataset> & train, const shared_ptr<dataset> & test, const vector<NumericType> &  test_params) = 0 ;

		virtual bool save_machine(const M & machine, const string & file_name) = 0;

		virtual bool save_performance(const blitz::Array<NumericType,3> & perf, const string & filename) = 0;

		virtual shared_ptr<M> load_machine(const string & file_name) = 0;



	public:

		void do_preprocess(bool process_)
		{
			preprocess = process_;
		}
		void set_param_candidate(const parameter_set & train_candidate_, const parameter_set & test_candidate_)
		{
			train_candidate = train_candidate_;
			test_candidate = test_candidate_;
		}

	protected:

		void log_experiment_configuration()
		{
			log->info(" The configuration of experiment : ");

			std::ostringstream info_ss;
			info_ss << " Training-testing pairs : " << datasets.get_train_test_pair_num();
			log->info(info_ss.str());
			info_ss.clear();info_ss.str("");

			info_ss << " Cross-validation folder nums for each training set : " ;
			for (int i = 0;i < datasets.get_train_test_pair_num();i++)
			{
				info_ss << datasets.get_train_test_pair(i).get_tv_folder_num() << " " ;
			}
			info_ss<< endl;

			log->info(info_ss.str());
			info_ss.clear();info_ss.str("");

			info_ss <<" There are " << train_candidate.get_param_num() << " training parameter candidates :" <<endl;
			for (int i = 0; i < train_candidate.get_param_num(); i ++ )
			{
				info_ss << "\t the candidates for " << i <<  " -th training parameter are :" ;
				const vector<NumericType> & cur_train_candidate = train_candidate.get_param_candidate(i);
				BOOST_FOREACH(NumericType x,cur_train_candidate){info_ss << x << " ";}
//				for_each(cur_train_candidate.begin(),cur_train_candidate.end(),[this](NumericType param){logfile << param << " " ;});
				info_ss << endl;
			}

			log->info(info_ss.str());
			info_ss.clear();info_ss.str("");

			info_ss << " There are " << test_candidate.get_param_num() <<  " test parameter candidates :" <<endl; 
			for (int i = 0; i < test_candidate.get_param_num(); i ++ )
			{
				info_ss << "\t the candidates for " << i <<  " -th testing parameter are :" ;
				const vector<NumericType> & cur_test_candidate = test_candidate.get_param_candidate(i);
				BOOST_FOREACH(NumericType x,cur_test_candidate){info_ss << x << " ";}
				//for_each(cur_test_candidate.begin(),cur_test_candidate.end(),[this](NumericType param){logfile << param << " " ;});
				info_ss << endl ;
			}

			log->info(info_ss.str());
			info_ss.clear();info_ss.str("");
		}

	

	public:

		virtual NumericType evaluate() = 0;
	
	};
}

#endif
