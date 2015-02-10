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
#include <boost/foreach.hpp>
#include <blitz/array.h>
#include <boost/algorithm/string/join.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include "experiment.h"
#include <iostream>

#ifdef USE_GPU 
#define MULTI_THREAD 0
#elif defined USE_PARTIAL_GPU
#define MULTI_THREAD 0
#else
#define MULTI_THREAD 1
#endif

#undef max
#undef min

namespace experiment
{
	// 	template <class M> class random_machine_experiment;

	// 	template <class M>
	// 	class train_test_thread_functor
	// 	{
	// 		random_machine_experiment<M> & experiment;
	// 		int m;
	// 		int j;
	// 		int id;
	// 		const train_validation_pair & cv_pair;
	// 		const vector<vector<NumericType>>& train_param_comb;
	// 		const vector<vector<NumericType>>& test_param_comb;
	// 		blitz::Array<NumericType, 3> & performance;
	// 	public:
	// 		train_test_thread_functor(random_machine_experiment<M> & experiment_, const train_validation_pair & cv_pair_,
	// 			const vector<vector<NumericType>>& train_param_comb_, const vector<vector<NumericType>>& test_param_comb_, 
	// 			blitz::Array<NumericType, 3> & performance_, int m_ , int j_, int id_)
	// 			:experiment(experiment_),cv_pair(cv_pair_),
	// 			train_param_comb(train_param_comb_),test_param_comb(test_param_comb_),
	// 			performance(performance_),m(m_),j(j_),id(id_)
	// 		{
	// 		}

	// 		tuple<shared_ptr<M>, double> operator()()
	// 		{
	// 			shared_ptr<dataset>  train = cv_pair.get_train_dataset();
	// 			shared_ptr<dataset>  valid = cv_pair.get_validation_dataset();

	// 			shared_ptr<dataset>  proc_train, proc_valid;
	// 			tie(proc_train,proc_valid) = experiment.prepare_dataset(train,valid);

	// 			shared_ptr<M> machine;
	// 			std::ostringstream sufix_ss;
	// 			sufix_ss << experiment.get_experiment_name() << "_machine_" << id << "_" << m << "_" << j;

	// 			string filename = sufix_ss.str();

	// 			if (experiment.load_existing)
	// 			{
	// 				machine = experiment.load_machine(filename);
	// 			}
	// 			else
	// 			{
	// 				machine = experiment.train_one_machine(proc_train,train_param_comb[m],proc_valid);

	// //				experiment.save_machine(* machine, filename );
	// 			}

	// 			if (test_param_comb.size() == 0)
	// 			{
	// 				vector<NumericType> empty_param;
	// 				NumericType cur_perf  = experiment.test_performance(*machine, proc_train, proc_valid, empty_param);
	// 				{
	// #if MULTI_THREAD
	// 					boost::mutex::scoped_lock lock(experiment.performance_update_mutex);
	// #endif
	// 					performance(m,0,j) = cur_perf;
	// 				}
	// 			}
	// 			else
	// 			{
	// 				for (int k = 0; k < test_param_comb.size(); k++)
	// 				{
	// 					NumericType cur_perf  = experiment.test_performance(*machine, proc_train, proc_valid, test_param_comb[k]);
	// 					{
	// 	#if MULTI_THREAD
	// 						boost::mutex::scoped_lock lock(experiment.performance_update_mutex);
	// 	#endif
	// 						performance(m,k,j) = cur_perf;
	// 						//experiment.logfile << "Validation performance of " << sufix_ss.str() <<  " is " << cur_perf << std::endl;
	// 					}
	// 				}
	// 			}
	// 		}
	// 	};

	template<typename M>
	class random_machine_experiment: public experiment<M>
	{
		//		friend class train_test_thread_functor<M>;
	protected:
		int try_num;

		int thread_num;

		boost::mutex performance_update_mutex;

		bool load_existing;

		using experiment<M>::log;
		using experiment<M>::train_candidate;
		using experiment<M>::test_candidate;

		using experiment<M>::load_machine;
		using experiment<M>::datasets;
		using experiment<M>::prepare_dataset;
		using experiment<M>::log_experiment_configuration;

		using experiment<M>::train_one_machine;
		using experiment<M>::test_performance;



	public:
		random_machine_experiment(const experiment_datasets & datasets_):experiment<M>(datasets_)
		{
			try_num = 1;
			load_existing = false;
		}
		~random_machine_experiment(void)
		{
		}

	public:

		void set_experiment_num(int num)
		{
			try_num = num;
		}

		void set_thread_num(int num)
		{
			thread_num = num;
		}

		void set_load_existing(bool load)
		{
			load_existing = load;
		}

	private:

		static string make_random_machine_ID(int datasetid, const vector<NumericType> & train_param, int randomind)
		{
			std::ostringstream sufix_ss;
			sufix_ss << platform::Instance().getExperimentID() << "_datasetid=" << datasetid << "_";

			sufix_ss << "trainparam=[" ;

			vector<string> train_param_str;

			BOOST_FOREACH(NumericType param, train_param)
			{
				train_param_str.push_back(boost::lexical_cast<string>(param));
			}

			sufix_ss << boost::algorithm::join(train_param_str,",");
			sufix_ss << "]_";

			sufix_ss << "randomind=" << randomind << ".machine";


			return  sufix_ss.str() ;
		}

		static string make_machine_ID(int datasetid)
		{
			std::ostringstream sufix_ss;
			sufix_ss << platform::Instance().getExperimentID() << "_datasetid=" << datasetid ;

			return  sufix_ss.str() ;
		}


		static string make_performance_ID( int randomind)
		{
			std::ostringstream sufix_ss;
			sufix_ss << platform::Instance().getExperimentID() << "_randomind="  << randomind << ".performance";

			return sufix_ss.str();
		}



		tuple<shared_ptr<M>, vector<NumericType>, vector<NumericType>, NumericType>  learn_a_machine(const train_validation_pair & cv_pair ,int id)
		{
			// get and print the current time

			std::ostringstream info_ss;
			//info_ss << "Training Started "<< std::endl;
			//log->info(info_ss.str());
			//info_ss.clear();info_ss.str("");

			vector<vector<NumericType>> train_param_comb = train_candidate.emurate_parameter_combination();
			vector<vector<NumericType>> test_param_comb = test_candidate.emurate_parameter_combination();

			blitz::Array<NumericType, 3> performance(train_param_comb.size(),std::max(int(test_param_comb.size()),1),try_num);
			//blitz::Array<shared_ptr<M>, 2> machines(train_param_comb.size(),try_num);


#if MULTI_THREAD
			boost::thread_group train_test_threads;
#endif
			for (int m = 0; m < train_param_comb.size(); m++)
			{

				for (int j = 0; j < try_num; j++)
				{
					//					train_test_thread_functor<M> functor(*this, cv_pair,train_param_comb, test_param_comb, 	performance, m , j, id);

					// #if MULTI_THREAD
					// 					if (thread_num == 1)
					// 					{
					// 						functor();
					// 					}
					// 					else
					// 					{
					// 						train_test_threads.create_thread(functor);
					// 					}
					// #else
					// 					functor();
					// #endif

#if MULTI_THREAD
					train_test_threads.create_thread([&, this,m, j ](){
#endif
						shared_ptr<dataset>  train = cv_pair.get_train_dataset();
						shared_ptr<dataset>  valid = cv_pair.get_validation_dataset();
						shared_ptr<M> machine;

						string machineID = random_machine_experiment<M>::make_random_machine_ID(id,train_param_comb[m],j);

						shared_ptr<dataset>  proc_train, proc_valid;
						tie(proc_train,proc_valid) = this->prepare_dataset(train,valid);

						if (this->load_existing)
						{
							machine = this->load_machine(machineID);
						}
						else
						{

							machine = this->train_one_machine(machineID,proc_train,train_param_comb[m],proc_valid);

							this->save_machine(* machine, machineID );
						}

						int k = 0;
						do
						{
							vector<NumericType> test_param;
							if (test_param_comb.size() != 0)
							{
								test_param = test_param_comb[k];
							}

							NumericType cur_perf  = this->test_performance(*machine, proc_train, proc_valid, test_param);
							{
#if MULTI_THREAD
								boost::mutex::scoped_lock lock(this->performance_update_mutex);
#endif
								performance(m,k,j) = cur_perf;
								//logfile << "Validation performance of " << sufix_ss.str() <<  " is " << cur_perf << std::endl;
							}
							k ++;
						}while (k < test_param_comb.size());

						// 						for (int k = 0; k < test_param_comb.size(); k++)
						// 						{
						// 							NumericType cur_perf  = test_performance(*machine, *proc_train, *proc_valid, test_param_comb[k]);
						// 							{
						// #if MULTI_THREAD
						// 								boost::mutex::scoped_lock lock(performance_update_mutex);
						// #endif
						// 								performance(m,k,j) = cur_perf;
						// 								//logfile << "Validation performance of " << sufix_ss.str() <<  " is " << cur_perf << std::endl;
						// 							}
						// 						}
#if MULTI_THREAD
					});
#endif
#if MULTI_THREAD
					if (train_test_threads.size() % thread_num == 0)
					{
						train_test_threads.join_all();

						//logfile << thread_num  << " netowrks are trained" << std::endl;
					}
#endif
				}

			}
#if MULTI_THREAD
			train_test_threads.join_all();
#endif


			//info_ss << "Training finished "<< std::endl;
			//log->info(info_ss.str());
			//info_ss.clear();info_ss.str("");

			string perf_file_name = random_machine_experiment::make_performance_ID(id);

			this->save_performance(performance,perf_file_name);

			NumericType validation_performance = blitz::max(performance);
			blitz::TinyVector<int,3> index = blitz::maxIndex(performance);

			int m = index(0);
			int j = index(2);

			string machineID = random_machine_experiment::make_random_machine_ID(id,train_param_comb[m],j);
			shared_ptr<M> best_machine = load_machine(machineID);


			vector<NumericType> best_train_param = train_param_comb[m];
			vector<NumericType> best_test_param;

			if (test_param_comb.size() > 0)
				best_test_param = test_param_comb[index(1)];

			return make_tuple(best_machine,best_train_param,best_test_param,validation_performance);

		}

		NumericType calculate_test_performance(const shared_ptr<M> & machine, const shared_ptr<dataset> & train, const shared_ptr<dataset> & test, const vector<NumericType> & optim_test_param)
		{
			NumericType performance = test_performance( *machine,train,test, optim_test_param);
			return performance;
		}

	public:

		virtual NumericType evaluate()
		{
			log_experiment_configuration();

			log->info( "Begining experiment: ");

			NumericType over_all_perf = 0;

			log->info( "------------------------------------------------" );

			NumericType perf = 0;

			std::ostringstream info_ss;


			for(int  j = 0; j < datasets.get_train_test_pair_num(); j++)
			{
				shared_ptr<M> machine;
				vector<NumericType> optim_train_params;
				vector<NumericType> optim_test_params;
				NumericType cur_valid_perf;
				tie(machine, optim_train_params, optim_test_params,cur_valid_perf) = learn_a_machine(datasets.get_train_test_pair(j).get_tv_pair(0), j);

				shared_ptr<dataset> train = datasets.get_train_test_pair(j).get_tv_pair(0).get_train_dataset();
				shared_ptr<dataset> test = datasets.get_train_test_pair(j).get_test_dataset();

				shared_ptr<dataset> proc_train, proc_test;
				tie(proc_train,proc_test) = prepare_dataset(train,test);


				NumericType cur_test_perf = calculate_test_performance(machine, proc_train, proc_test,  optim_test_params);



				info_ss << "\t" << "The experimental result for " << j <<"-th train-test pair: ";
				log->info(info_ss.str());
				info_ss.clear();info_ss.str("");

				info_ss << "\t" << "\t The optimal training params are : " ;
				BOOST_FOREACH(NumericType param,optim_train_params){info_ss << param << " ";}
				//				for_each(optim_train_params.begin(),optim_train_params.end(),[this](NumericType param){logfile << param << " " ;});
				log->info(info_ss.str());
				info_ss.clear();info_ss.str("");
				info_ss << "\t" << "\t The optimal testing params are : " ;
				BOOST_FOREACH(NumericType param,optim_test_params){info_ss << param << " ";}
				//				for_each(optim_test_params.begin(),optim_test_params.end(),[this](NumericType param){logfile << param << " " ;});
				log->info(info_ss.str());
				info_ss.clear();info_ss.str("");
				info_ss << "\t" << "\t The best validation performance is : " << cur_valid_perf ; 
				log->info(info_ss.str());
				info_ss.clear();info_ss.str("");
				info_ss << "\t" << "\t The test performance is : " << cur_test_perf ; 
				log->info(info_ss.str());
				info_ss.clear();info_ss.str("");




				over_all_perf = over_all_perf + cur_test_perf;
				std::ostringstream sufix_ss;
				sufix_ss <<  j;

				string machineID = random_machine_experiment::make_machine_ID(j);
				save_machine(* machine, machineID + ".best" );

				//shared_ptr<M> machine2 = deserialize_from_hdf_file<M>(filename + ".hdf");

				//NumericType test_perf_2 = calculate_test_performance(machine2, *proc_train, *proc_test,  optim_test_params);

				//if (abs(test_perf_2 - cur_test_perf) > 1e-6)
				//	throw runtime_error("machine serialize error!");

			}


			over_all_perf = over_all_perf/datasets.get_train_test_pair_num();

			info_ss << "The average testing performance for all train test pairs is : " << over_all_perf << endl;
			log->info(info_ss.str());
			info_ss.clear();info_ss.str("");
			log->info("The experiment is finished.") ;

			return over_all_perf;

		}
	};
}
#endif
