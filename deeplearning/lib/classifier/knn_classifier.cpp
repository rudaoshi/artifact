#include <liblearning/classifier/knn_classifier.h>
#include <liblearning/util/algo_util.h>

#include <algorithm>

using namespace classification;
knn_classifier::knn_classifier(const supervised_dataset & train_, int k_):train(train_),k(k_)
{

}


knn_classifier::~knn_classifier(void)
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

NumericType knn_classifier::test(const supervised_dataset & test)
{
	vector<int> test_label(test.get_sample_num());
	if (train.get_sample_num() * test.get_sample_num() > 5000* 5000)
	{
		const MatrixType & test_data = test.get_data();
		const MatrixType & train_data = train.get_data();
		for (int i = 0;i<test.get_sample_num();i++)
		{
			vector<NumericType> cur_dist(train.get_sample_num());
			for (int j = 0; j<train.get_sample_num();j++)
			{
				cur_dist[j] = (train_data.col(j)-test_data.col(i)).squaredNorm();
			}

			vector<unsigned int> index =  nth_element_index(cur_dist.begin(),cur_dist.begin()+k,cur_dist.end());

			vector<int> class_labels(k);

			std::transform(index.begin(),index.end(),class_labels.begin(),
			 [&](unsigned int n)-> int
			  {
				  return train.get_label()[n];
			  }
			);
			//std::transform(index.begin(),index.end(),class_labels.begin(),index2label_functor(train.get_label()));


			vector<int> elem_nums(k);
			std::transform(class_labels.begin(),class_labels.end(),elem_nums.begin(),
			 [&class_labels](int n)-> int
			  {
				  return count(class_labels.begin(),class_labels.end(),n);
			  }
			);

			//std::transform(class_labels.begin(),class_labels.end(),elem_nums.begin(),Count(class_labels));

			vector<int>::iterator max_pos = std::max_element(elem_nums.begin(),elem_nums.end());

			test_label[i] = class_labels[max_pos - elem_nums.begin()];

		}
	}
	else
	{
		MatrixType dist = sqdist(train.get_data(),test.get_data());

		for (int i = 0;i<dist.cols();i++)
		{
			EigenVectorType cur_dist= (EigenVectorType)dist.col(i);

			vector<unsigned int> index =  nth_element_index(cur_dist.data(),cur_dist.data()+k,cur_dist.data()+cur_dist.size());

			vector<int> class_labels(k);

			std::transform(index.begin(),index.end(),class_labels.begin(),
			 [&](unsigned int n)-> int
			  {
				  return train.get_label()[n];
			  }
			);
			//std::transform(index.begin(),index.end(),class_labels.begin(),index2label_functor(train.get_label()));


			vector<int> elem_nums(k);
			std::transform(class_labels.begin(),class_labels.end(),elem_nums.begin(),
			 [&class_labels](int n)-> int
			  {
				  return count(class_labels.begin(),class_labels.end(),n);
			  }
			);

			//std::transform(class_labels.begin(),class_labels.end(),elem_nums.begin(),Count(class_labels));

			vector<int>::iterator max_pos = std::max_element(elem_nums.begin(),elem_nums.end());

			test_label[i] = class_labels[max_pos - elem_nums.begin()];
		}
	}
	int correctNum = 0;

	for (int i =0; i<test.get_sample_num();i++)
	{
		if (test_label[i] == test.get_label()[i])
		{
			correctNum ++;
		}
	}


	return NumericType(correctNum)/test.get_sample_num();

}