#ifndef NEAREST_CENTER_CLASSIFIER_H
#define NEAREST_CENTER_CLASSIFIER_H

#include <liblearning/core/classifier.h>
#include <liblearning/core/maker.h>
#include <liblearning/core/supervised_dataset.h>


namespace classification
{
	using namespace core;

	
	/* 
	struct NearestCenterClassifierParam
	{
		shared_ptr<kernel> kernelfunc;
		double gamma;
	}; 
	*/
	
	
	class NearestCenterClassifier : public classifier
	{
		
	public:
		
		shared_ptr<supervised_dataset> train_set;
		
		MatrixType centers;
		
	public:
		NearestCenterClassifier( );
		~NearestCenterClassifier();
		
		const MatrixType & get_center();
		
		virtual void train(const shared_ptr<dataset>  &);
		
		virtual NumericType test(const shared_ptr<dataset>  &, const shared_ptr<dataset>  &);
	};
	
}

#endif


