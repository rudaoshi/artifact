#ifndef DATASET_DESIGN_H
#define DATASET_DESIGN_H


#include <liblearning/core/dataset.h>

#include <vector>
using namespace std;

#include "train_test_pair.h"
namespace experiment
{
class experiment_datasets: public direct_xml_file_serializable, public direct_hdf_file_serializable
{

protected:

	vector<train_test_pair> train_test_pairs;

public:
	experiment_datasets(void);
	~experiment_datasets(void);

	void make_folder_based_train_test_pairs(const dataset & data,const dataset_splitter & splitter);
	void make_random_splitted_train_test_pairs(const dataset & data,const dataset_splitter & splitter, int pair_num);
	void set_train_test_pairs(const dataset & train, const dataset & test,int pair_num);

	void prepare_cross_validation(const dataset_splitter & splitter);
	void random_split_train_validation(const dataset_splitter & splitter);

	int get_train_test_pair_num() const;

	const train_test_pair &  get_train_test_pair(int i ) const;
private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & BOOST_SERIALIZATION_NVP(train_test_pairs);
	}
public:

	virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

	virtual void decode_xml_node(rapidxml::xml_node<> & node);

	// HDF5 serialization

	virtual void encode_hdf_node(H5::Group * obj) const;

	virtual void decode_hdf_node(const H5::Group * obj) ;

};
}
CAMP_TYPE(experiment::experiment_datasets)
#endif /* DATASET_DESIGN */
