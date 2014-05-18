#ifndef TRAIN_TEST_DATA_DESIGN_H
#define TRAIN_TEST_DATA_DESIGN_H

#include <liblearning/core/dataset.h>
#include "train_validation_pair.h"
#include <liblearning/core/data_splitter.h>

namespace experiment
{
	using namespace core;

class train_test_pair: public xml_serializable, public hdf_serializable
{


protected:

	shared_ptr<dataset> test;

	shared_ptr<dataset> train;

	vector<train_validation_pair > train_validation_pairs;
	
public:

	train_test_pair();

	train_test_pair(const shared_ptr<dataset>& train, const shared_ptr<dataset>& test);

	~train_test_pair(void);

	const shared_ptr<dataset> & get_train_dataset() const;
	const shared_ptr<dataset> & get_test_dataset() const;

	int get_tv_folder_num() const ;

	void make_cross_validation_pairs(const dataset_splitter & splitter);

	void make_random_train_validation_pairs(const dataset_splitter & splitter);

	const train_validation_pair & get_tv_pair(int i) const ;

	const vector<train_validation_pair > & get_all_tv_pairs() const;

private:
	friend class boost::serialization::access;

	template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
		dataset * p_train = train.get();
		dataset * p_test = test.get();

		ar & boost::serialization::make_nvp("train_set",p_train);
		ar & boost::serialization::make_nvp("test_set",p_test);
		ar & boost::serialization::make_nvp("train_validation_sets",train_validation_pairs);

    }
    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
		dataset * p_train;
		dataset * p_test;

		ar & boost::serialization::make_nvp("train_set",p_train);
		ar & boost::serialization::make_nvp("test_set",p_test);

		train.reset(p_train);
		test.reset(p_test);

		ar & boost::serialization::make_nvp("train_validation_sets",train_validation_pairs);

    }
    BOOST_SERIALIZATION_SPLIT_MEMBER();

public:

	virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

	virtual void decode_xml_node(rapidxml::xml_node<> & node);

	// HDF5 serialization

	virtual void encode_hdf_node(H5::Group * obj) const;

	virtual void decode_hdf_node(const H5::Group * obj) ;
};

}
CAMP_TYPE(experiment::train_test_pair)
#endif