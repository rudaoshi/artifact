#ifndef CROSS_VALIDATION_EXPERIMENT_H
#define CROSS_VALIDATION_EXPERIMENT_H

#include "experiment.h"

namespace experiment
{
	template<typename M>
	class POCO_EXPORT cross_validation_experiment: public experiment<M>
	{

	private:

		tuple<vector<NumericType>, vector<NumericType>,NumericType > select_param_one_pair(const vector<train_validation_pair > & cv_pairs)
		{

			vector<vector<NumericType>> train_param_comb = train_candidate.emurate_parameter_combination();
			vector<vector<NumericType>> test_param_comb = test_candidate.emurate_parameter_combination();

			MatrixType performance = MatrixType::Zero(train_param_comb.size(),test_param_comb.size());

			boost::thread_group train_test_threads;

			boost::mutex performance_update_mutex;

			for (int m = 0; m < train_param_comb.size(); m++)
			{

				for (int j = 0; j < cv_pairs.size(); j++)
				{
					train_test_threads.create_thread([&, m,j ](){
						shared_ptr<dataset>  train = cv_pairs[j].get_train_dataset();
						shared_ptr<dataset>  valid = cv_pairs[j].get_validation_dataset();

						shared_ptr<dataset>  proc_train, proc_valid;
						tie(proc_train,proc_valid) = prepare_dataset(*train,*valid);

						shared_ptr<M> machine = train_one_machine(*proc_train,train_param_comb[m]);

						for (int k = 0; k < test_param_comb.size(); k++)
						{
							NumericType cur_perf  = test_performance(*machine, *proc_train, *proc_valid, test_param_comb[k]);

							{
								boost::mutex::scoped_lock lock(performance_update_mutex);
								performance(m,k) += cur_perf;
							}
						}

					});
				}

			}

			train_test_threads.join_all();

			auto abs_max_pos = max_element(performance.data(), performance.data()+performance.size());
			int relative_max_pos = abs_max_pos - performance.data();

			// performance (in type of MaxtrixXd) is column major
			int row = relative_max_pos% performance.rows();
			int col = relative_max_pos/ performance.rows();

			NumericType validation_performance = performance(row,col)/cv_pairs.size();

			return make_tuple(train_param_comb[row],test_param_comb[col],validation_performance);

		}

		tuple<NumericType, shared_ptr<M>> calculate_test_performance(const dataset & train, const dataset & test, const vector<NumericType> & optim_train_param, const vector<NumericType> & optim_test_param)
		{
			shared_ptr<M> machine = train_one_machine(train,optim_train_param);
			NumericType performance =   test_performance( *machine,train,test, optim_test_param);
			return make_tuple(performance, machine);
		}

	public:

		NumericType evaluate()
		{
			log_experiment_configuration();

			logfile << "Begining experiment: " << endl;

			NumericType over_all_perf = 0;

			NumericType perf = 0;

			for(int  j = 0; j < datasets.get_train_test_pair_num(); j++)
			{

				vector<NumericType> optim_train_params, optim_test_params;
				NumericType cur_valid_perf;
				tie(optim_train_params, optim_test_params,cur_valid_perf) = select_param_one_pair(datasets.get_train_test_pair(j).get_all_cv_pairs());

				shared_ptr<dataset> train = datasets.get_train_test_pair(j).get_train_dataset();
				shared_ptr<dataset> test = datasets.get_train_test_pair(j).get_test_dataset();

				shared_ptr<dataset> proc_train, proc_test;
				tie(proc_train,proc_test) = prepare_dataset(*train,*test);

				shared_ptr<M> machine;
				NumericType cur_test_perf;
				tie(cur_test_perf,machine) = calculate_test_performance(*proc_train, *proc_test,  optim_train_params, optim_test_params);

				logfile << "\t" << "The experimental result for " << j <<"-th train-test pair: " <<endl;
				logfile << "\t" << "\t The optimal training params are : " ;
				for_each(optim_train_params.begin(),optim_train_params.end(),[this](NumericType param){logfile << param << " " ;});
				logfile << endl;
				logfile << "\t" << "\t The optimal testing params are : " ;
				for_each(optim_test_params.begin(),optim_test_params.end(),[this](NumericType param){logfile << param << " " ;});
				logfile << endl;
				logfile << "\t" << "\t The best average validation performance is : " << cur_valid_perf << endl; 
				logfile << "\t" << "\t The test performance is : " << cur_test_perf << endl; 
				logfile << endl;

				logfile.flush();


				over_all_perf = over_all_perf + cur_test_perf;
				std::ostringstream sufix_ss;
				sufix_ss << i << "_" << j;
				save_machine(* machine, experiment_name + sufix_ss.str() );
			}


			over_all_perf = over_all_perf/datasets.get_train_test_pair_num();

			logfile << "The average testing performance for all experiments is : " << over_all_perf << endl;

			logfile << "The experiment is finished." << endl;

			return over_all_perf;

		}
	};

}
#endif