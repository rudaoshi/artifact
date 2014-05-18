
#ifndef EVALUATOR_IMPL_H
#define EVALUATOR_IMPL_H


class FeatureExtractorEvaluator : public evaluator
{
	parameter_set   test_candidate;

	shared_ptr<IDataSet>  train_set; 
	shared_ptr<IDataSet>  test_set;

  NumericType globalBestPerf;

  log4cpp::Category * log;

  string evaluatorName;

public:

	deep_auto_encoder_performance_evaluator(const shared_ptr<supervised_dataset> & train_set_, 
						const shared_ptr<supervised_dataset> & test_set_, const parameter_set &  test_candidate_, const string & evaluatorName_):train_set(train_set_),test_set(test_set_),test_candidate(test_candidate_),evaluatorName(evaluatorName_)
	{
		globalBestPerf = 0;
	}

protected:
	virtual double test_performance_on_one_classifier(const shared_ptr<supervised_dataset> & p_train_feature, 
		const shared_ptr<supervised_dataset> & p_test_feature, const vector<NumericType> & cur_test_param) = 0;

public:
  virtual double evaluate( deep_auto_encoder & net, vector<NumericType> & optim_classifier_param)
	{

	  log = & platform::Instance().get_log(Temp,evaluatorName);

		const supervised_dataset  & s_train = dynamic_cast<const supervised_dataset  &>(*train_set);
		const supervised_dataset  & s_test = dynamic_cast<const supervised_dataset  &>(*test_set);

		shared_ptr<supervised_dataset> p_train_feature = dynamic_pointer_cast<supervised_dataset>(net.encode(s_train));

		shared_ptr<supervised_dataset>  p_test_feature = dynamic_pointer_cast<supervised_dataset>(net.encode(s_test));

		vector<vector<NumericType>> test_param_comb = test_candidate.emurate_parameter_combination();

		double best_perf = 0;

		if (test_param_comb.size() == 0)
		{
			vector<double> empty_param_set;
			best_perf = test_performance_on_one_classifier(p_train_feature,p_test_feature,empty_param_set);
			optim_classifier_param =  empty_param_set;
		}
		else
		{
			for (int k = 0; k < test_param_comb.size(); k++)
			{
			
				double cur_perf = test_performance_on_one_classifier(p_train_feature,p_test_feature,test_param_comb[k]);

				if (best_perf < cur_perf)
				{
					best_perf = cur_perf;
					optim_classifier_param = test_param_comb[k];
				}
			}
		}


		if (best_perf > globalBestPerf)
		{
			globalBestPerf = best_perf;

			std::ostringstream info_ss;

			info_ss << "[" << net.getMachineID() << "]" << " Machine with higher performance found at iteration : " << net.get_finetune_running_iter() << "! Current Best Perf = "<< best_perf << "!";
			
			if (optim_classifier_param.size() > 0)
			{
				info_ss << "Optimal param:";
				
				for (int i = 0;i < optim_classifier_param.size(); i++)
				{
					info_ss << optim_classifier_param[i] ;
					if (i < optim_classifier_param.size() -1) 
					  info_ss << ",";
				}
			}
			log->info(info_ss.str());
			//path best_machine_file_path = platform::Instance().getCurrentTempDirPath() / path(net.getMachineID() + "." + evaluatorName + ".hdf"); 
			//boost::filesystem::remove(boost::filesystem::path(best_machine_file_path));

			//net.save_hdf(best_machine_file_path.string());

			//log->info( "Current best machine has been saved to  " + best_machine_file_path.string() );
		}
		//else if (best_perf == globalBestPerf)
		//{
		//  std::ostringstream info_ss;
		//  info_ss <<  "Performance remains not changed! Current Perf = "<< best_perf << "! ";
		//  log->info(info_ss.str());	
		//}
		//else 
		//{
		//  std::ostringstream info_ss;
		//  info_ss <<  "Performance Dropped! Current Perf = "<< best_perf << "Current Best Perf = "<< globalBestPerf <<"!";
		//  log->info(info_ss.str());	
		//}

		return best_perf;

	}
};


class KNN_performance_evaluator : public deep_auto_encoder_performance_evaluator
{
	

public:

	KNN_performance_evaluator( const shared_ptr<supervised_dataset> & train_set_, 
				const shared_ptr<supervised_dataset> & test_set_, const parameter_set &  test_candidate_)
	  :deep_auto_encoder_performance_evaluator(train_set_,test_set_,test_candidate_,"knn")
	{

	}
	
protected:

	virtual double test_performance_on_one_classifier(const shared_ptr<supervised_dataset> & p_train_feature, 
		const shared_ptr<supervised_dataset> & p_test_feature, const vector<NumericType> & cur_test_param)
	{
		knn_classifier clf(*p_train_feature, cur_test_param[0]);

		return clf.test(*p_test_feature);
	}

};

class nc_performance_evaluator : public deep_auto_encoder_performance_evaluator
{
	

public:

	nc_performance_evaluator( const shared_ptr<supervised_dataset> & train_set_, 
				const shared_ptr<supervised_dataset> & test_set_, const parameter_set &  test_candidate_)
	  :deep_auto_encoder_performance_evaluator(train_set_,test_set_,test_candidate_,"nc")
	{

	}
	
protected:

	virtual double test_performance_on_one_classifier(const shared_ptr<supervised_dataset> & p_train_feature, 
		const shared_ptr<supervised_dataset> & p_test_feature, const vector<NumericType> & cur_test_param)
	{
		NearestCenterClassifier ncf;
		ncf.train(p_train_feature);
		return ncf.test(p_train_feature,p_test_feature);
	}

};

class svm_performance_evaluator : public deep_auto_encoder_performance_evaluator
{
	

public:

	svm_performance_evaluator( const shared_ptr<supervised_dataset> & train_set_, 
				const shared_ptr<supervised_dataset> & test_set_, const parameter_set &  test_candidate_)
	  :deep_auto_encoder_performance_evaluator(train_set_,test_set_,test_candidate_,"svm")
	{

	}
protected:
	double test_performance_on_one_classifier(const shared_ptr<supervised_dataset> & p_train_feature, 
		const shared_ptr<supervised_dataset> & p_test_feature, const vector<NumericType> & cur_test_param)
	{
		unit_interval_transform preprocessor(*p_train_feature);

		shared_ptr<dataset> proc_train = preprocessor.apply(*p_train_feature) ;
		shared_ptr<dataset> proc_test = preprocessor.apply(*p_test_feature) ;
		LibSVMParam param;
		param.kernelfunc.reset(new kernelmethod::linear_kernel);
		param.C = cur_test_param[0];

		LibSVMClassifier svmclf;
		svmclf.make(param);
		svmclf.train(proc_train);
		return svmclf.test(proc_train,proc_test);
		
	}

};

class linear_svm_performance_evaluator : public deep_auto_encoder_performance_evaluator
{
	

public:

	linear_svm_performance_evaluator( const shared_ptr<supervised_dataset> & train_set_, 
				const shared_ptr<supervised_dataset> & test_set_, const parameter_set &  test_candidate_)
	  :deep_auto_encoder_performance_evaluator(train_set_,test_set_,test_candidate_,"lsvm")
	{

	}
protected:
	virtual double test_performance_on_one_classifier(const shared_ptr<supervised_dataset> & p_train_feature, 
		const shared_ptr<supervised_dataset> & p_test_feature, const vector<NumericType> & cur_test_param)
	{

		unit_interval_transform preprocessor(*p_train_feature);

		shared_ptr<dataset> proc_train = preprocessor.apply(*p_train_feature) ;
		shared_ptr<dataset> proc_test = preprocessor.apply(*p_test_feature) ;

		LinearSVMParam param;
		param.C = cur_test_param[0];

		LinearSVMClassifier svmclf;
		svmclf.make(param);
		svmclf.train(proc_train);
		return svmclf.test(proc_train,proc_test);

	}

};


class ldann_performance_evaluator : public deep_auto_encoder_performance_evaluator
{
	

public:

	ldann_performance_evaluator( const shared_ptr<supervised_dataset> & train_set_, 
				const shared_ptr<supervised_dataset> & test_set_, const parameter_set &  test_candidate_)
	  :deep_auto_encoder_performance_evaluator(train_set_,test_set_,test_candidate_,"ldann")
	{

	}
protected:
	virtual double test_performance_on_one_classifier(const shared_ptr<supervised_dataset> & p_train_feature, 
		const shared_ptr<supervised_dataset> & p_test_feature, const vector<NumericType> & cur_test_param)
	{
		 LDAParam param;

		 param.reg = cur_test_param[0];
		 
		 LDA lda;
		 lda.make(param); 
		 lda.train(p_train_feature);

		 shared_ptr<supervised_dataset> train_feature = dynamic_pointer_cast<supervised_dataset>(lda.extract_feature(p_train_feature));
		 shared_ptr<supervised_dataset> test_feature = dynamic_pointer_cast<supervised_dataset>(lda.extract_feature(p_test_feature));

		 knn_classifier clf(*train_feature, 1);

		 return clf.test(*test_feature);

	}

};
