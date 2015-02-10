#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include "machine.h"

namespace core
{
	template class evaluator
	{
		public:

			virtual void evaluate(machine & m) = 0;
	};

	//template <typename FeatrueExtractorParam, typename ClassifierParam> class supervised_feature_extractor_evaluator
	//	: public performance_evaluator<FeatrueExtractorParam, double>
	//{

	//	shared_ptr<supervised_dataset>  train_set, 
	//	shared_ptr<supervised_dataset>  test_set, 
	//	shared_ptr<classifier<ClassifierParam>>  classifier;

	//	public:

	//		feature_extractor_evaluator(const shared_ptr<supervised_dataset> & train_set_, 
	//			const shared_ptr<supervised_dataset> & test_set_, 
	//			const shared_ptr<classifier<ClassifierParam>> & classifier_)
	//			:train_set(train_set_),test_set(test_set_),classifier(classifier_)
	//		{

	//		}

	//		virtual double evaluate(const machine<MachineParam> & machine)
	//		{
	//			const feature_extractor<MachineParam> & feature_extractor = dynamic_cast<feature_extractor<MachineParam>>(machine);

	//			shared_ptr<data_set> train_feature = feature_extractor->extract_feature(train_set);
	//			shared_ptr<data_set> test_feature = feature_extractor->extract_feature(test_set);

	//			classifer->train(train_feature);

	//			return classifier->test(train_feature,test_feature);
	//			
	//		}
	//};
}
#endif
