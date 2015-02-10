/*
 *  NearestCenterClassifier.cpp
 *  learning
 *
 *  Created by Sun on 11-6-5.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <liblearning/classifier/NearestCenterClassifier.h>


#include <liblearning/util/algo_util.h>

#include <algorithm>

using namespace classification;
NearestCenterClassifier::NearestCenterClassifier()
{
	
}


NearestCenterClassifier::~NearestCenterClassifier()
{
}

#include <liblearning/util/matrix_util.h>

/*
 class index2label_functor
 {
 const vector<int> & label;
 public:
 
 index2label_functor(const vector<int> & label_):label(label_)
 {
 }
 
 int operator() (int n) 
 {
 return label[n];
 }
 
 };
 
 class Count
 {
 const vector<int> & label;
 public:
 
 Count(const vector<int> & label_):label(label_)
 {
 }
 
 int operator() (int n) 
 {
 return count(label.begin(),label.end(),n);;
 }
 };
 
 */


const MatrixType & NearestCenterClassifier::get_center()
{
	return centers;
}
 void NearestCenterClassifier::train(const shared_ptr<dataset>  & train)
{
	train_set = dynamic_pointer_cast<supervised_dataset>( train);
	
	if (!train_set)
	{
		throw new runtime_error("Nearest Center Classifier can only deal with supervised problems!");
	}
	
	centers = MatrixType::Zero(train_set->get_dim(),train_set->get_class_num());
	
	auto class_ids = train_set->get_class_id();
	auto class_elem_nums = train_set->get_class_elem_num();
	auto class_labels = train_set->get_label();
	auto samples = train_set->get_data();
	
	
	for (int i = 0; i < train_set->get_class_num(); i++) 
	{
		for (int j = 0; j < train_set->get_sample_num(); j++) 
		{
			if (class_labels[j] == class_ids[i]) 
			{
				centers.col(i) += samples.col(j);
			}
		}
		
		centers.col(i) /= class_elem_nums[i];
	}
	
}

NumericType NearestCenterClassifier::test(const shared_ptr<dataset>  & , const shared_ptr<dataset>  & test_)
{
	
	
	shared_ptr<supervised_dataset>  test_set = dynamic_pointer_cast<supervised_dataset >( test_);
	
	if (!test_set)
	{
		throw new runtime_error("Nearest Center Classifier can only deal with supervised problems!");
	}
	
	auto class_ids = train_set->get_class_id();
	
	vector<int> test_label(test_set->get_sample_num());

	MatrixType dist = sqdist(centers,test_set->get_data());
		
	for (int i = 0;i<dist.cols();i++)
	{
		EigenVectorType cur_dist= (EigenVectorType)dist.col(i);
			
		auto min_iter =  min_element(cur_dist.data(),cur_dist.data()+cur_dist.size());
			
		test_label[i] = class_ids[min_iter - cur_dist.data()];
			
			
	}
	
	int correctNum = 0;
	
	for (int i =0; i<test_set->get_sample_num();i++)
	{
		if (test_label[i] == test_set->get_label()[i])
		{
			correctNum ++;
		}
	}
	
	
	return NumericType(correctNum)/test_set->get_sample_num();
	
}

