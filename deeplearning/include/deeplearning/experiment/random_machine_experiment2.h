#ifndef RANDOM_MACHINE_EXPERIMENT_H_
#define RANDOM_MACHINE_EXPERIMENT_H_

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


#include <blitz/array.h>

#include "experiment2.h"
#include <liblearning/core/recognition_system.h>


namespace experiment
{

	using namespace core;

	template <typename FE, typename CL>
	class POCO_EXPORT random_machine_experiment2: public experiment2<recognition_system<FE,CL>>
	{

	protected:
		int try_num;

	public:
		random_machine_experiment2(const experiment_datasets & datasets_, const string & experiment_name_):experiment2(datasets_,experiment_name_)
		{

		}
		~random_machine_experiment2(void)
		{
		}

	public:

		void set_experiment_num(int num)
		{
			try_num = num;
		}

	private:


		tuple<shared_ptr<recognition_system<FE,CL> >, vector<NumericType>, NumericType>  learn_a_machine(const train_validation_pair & cv_pair)
		{

			vector<vector<NumericType>> param_comb = params.emurate_parameter_combination();

			blitz::Array<NumericType, 2> performance(param_comb.size(),try_num);
			blitz::Array<shared_ptr<recognition_system<FE,CL>>, 2> machines(param_comb.size(),try_num);

			

			boost::mutex performance_update_mutex;

			for (int m = 0; m < param_comb.size(); m++)
			{
				boost::thread_group eval_threads;
				for (int j = 0; j < try_num; j++)
				{
					

					eval_threads.create_thread([&, m, j ](){
						shared_ptr<dataset>  train = cv_pair.get_train_dataset();
						shared_ptr<dataset>  valid = cv_pair.get_validation_dataset();

						shared_ptr<dataset>  proc_train, proc_valid;
						tie(proc_train,proc_valid) = prepare_dataset(train,valid);

						machines(m,j) = train_one_machine(proc_train,param_comb[m]);

						{
							boost::mutex::scoped_lock lock(performance_update_mutex);
							performance(m,j) = machines(m,j)->test(proc_train, proc_valid);
						}

					});

				}
				eval_threads.join_all();
			}

			

			NumericType validation_performance = blitz::max(performance);
			auto index = blitz::maxIndex(performance);

			auto best_machine = machines(index(0),index(1));

			auto best_param = param_comb[index(0)];

			return make_tuple(best_machine,best_param,validation_performance);

		}

		//NumericType calculate_test_performance(const shared_ptr<M> & machine, const dataset & train, const dataset & test, const vector<NumericType> & optim_test_param)
		//{
		//	NumericType performance = test_performance( *machine,train,test, optim_test_param);
		//	return performance;
		//}

	public:

		virtual NumericType evaluate()
		{
			log_experiment_configuration();

			logfile << "Begining experiment: " << endl;

			NumericType over_all_perf = 0;

			logfile << "------------------------------------------------" << endl;

			NumericType perf = 0;

			for(int  j = 0; j < datasets.get_train_test_pair_num(); j++)
			{

				shared_ptr<recognition_system<FE,CL>> machine;
				vector<NumericType> optim_params;
				NumericType cur_valid_perf;
				tie(machine, optim_params, cur_valid_perf) = learn_a_machine(datasets.get_train_test_pair(j).get_tv_pair(0));

				shared_ptr<dataset> train = datasets.get_train_test_pair(j).get_train_dataset();
				shared_ptr<dataset> test = datasets.get_train_test_pair(j).get_test_dataset();

				shared_ptr<dataset> proc_train, proc_test;
				tie(proc_train,proc_test) = prepare_dataset(train,test);


				NumericType cur_test_perf = machine->test(proc_train, proc_test);

				logfile << "\t" << "The experimental result for " << j <<"-th train-test pair: " <<endl;
				logfile << "\t" << "\t The optimal params are : " ;
				for_each(optim_params.begin(),optim_params.end(),[this](NumericType param){logfile << param << " " ;});
				logfile << endl;

				logfile << "\t" << "\t The best validation performance is : " << cur_valid_perf << endl; 
				logfile << "\t" << "\t The test performance is : " << cur_test_perf << endl; 
				logfile << endl;

				logfile.flush();


				over_all_perf = over_all_perf + cur_test_perf;
				//std::ostringstream sufix_ss;
				//sufix_ss <<  j;
				//save_machine(* machine, experiment_name + sufix_ss.str() );
			}


			over_all_perf = over_all_perf/datasets.get_train_test_pair_num();

			logfile << "The average testing performance for all train test pairs is : " << over_all_perf << endl;

			logfile << "The experiment is finished." << endl;

			return over_all_perf;

		}
	};
}
#endif
