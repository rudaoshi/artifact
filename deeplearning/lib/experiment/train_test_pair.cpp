#include <liblearning/experiment/train_test_pair.h>

#include <liblearning/core/dataset_group.h>

using namespace experiment;

train_test_pair::train_test_pair()
{
}

train_test_pair::train_test_pair(	const shared_ptr<dataset>& train_, const shared_ptr<dataset>& test_):train(train_),test(test_)
{
}


train_test_pair::~train_test_pair(void)
{
}



void train_test_pair::make_cross_validation_pairs(const dataset_splitter & splitter)
{
	train_validation_pairs.erase(train_validation_pairs.begin(),train_validation_pairs.end());


	dataset_group group = splitter.split(*train);

	for (int j = 0;j<group.get_dataset_num();j++)
	{
		shared_ptr<dataset> p_cv_valid_set = group.get_dataset(j);

		shared_ptr<dataset> p_cv_train_set;

		for (int k = 0;k < group.get_dataset_num();k++)
		{
			if (k  != j)
			{
				if (!p_cv_train_set)
					p_cv_train_set = group.get_dataset(k)->clone();
				else
					p_cv_train_set->append(*group.get_dataset(k));
			}
		}

		train_validation_pairs.push_back(train_validation_pair(p_cv_train_set,p_cv_valid_set));
	}

}


void train_test_pair::make_random_train_validation_pairs(const dataset_splitter & splitter)
{
	train_validation_pairs.erase(train_validation_pairs.begin(),train_validation_pairs.end());

	dataset_group group = splitter.split(*train);

	if (group.get_dataset_num() != 2)
		throw std::runtime_error("The splitter do not split data set into two parts!");

	shared_ptr<dataset> p_train_set = group.get_dataset(0);

	shared_ptr<dataset> p_validation_set = group.get_dataset(1);
	train_validation_pairs.push_back(train_validation_pair(p_train_set,p_validation_set));
	
}

const shared_ptr<dataset> & train_test_pair::get_train_dataset() const
{
	return train;
}
const shared_ptr<dataset> & train_test_pair::get_test_dataset() const
{
	return test;
}

int train_test_pair::get_tv_folder_num() const
{
	return train_validation_pairs.size();
}

const train_validation_pair & train_test_pair::get_tv_pair(int i) const
{
	return train_validation_pairs[i];
}

const vector<train_validation_pair > & train_test_pair::get_all_tv_pairs() const
{

	return train_validation_pairs;
}



rapidxml::xml_node<> * train_test_pair::encode_xml_node(rapidxml::xml_document<> & doc) const
{
	using namespace rapidxml;


	char * train_test_pair_name = doc.allocate_string("train_test_pair"); 
	xml_node<> * train_test_pair_node = doc.allocate_node(node_element, train_test_pair_name);

	char * train_name = doc.allocate_string("train_set"); 
	xml_node<> * train_node = doc.allocate_node(node_element, train_name);
	train_node->append_node(train->encode_xml_node(doc));

	char * test_name = doc.allocate_string("test_set"); 
	xml_node<> * test_node = doc.allocate_node(node_element, test_name);
	test_node->append_node(test->encode_xml_node(doc));

	char * train_validation_pairs_name = doc.allocate_string("train_validation_pairs"); 
	xml_node<> * train_validation_pairs_node = doc.allocate_node(node_element, train_validation_pairs_name);

	for(int i = 0;i < train_validation_pairs.size();i++)
	{
		train_validation_pairs_node->append_node(train_validation_pairs[i].encode_xml_node(doc));
	}

	train_test_pair_node->append_node(train_node);
	train_test_pair_node->append_node(test_node);
	train_test_pair_node->append_node(train_validation_pairs_node);

	return train_test_pair_node;
}

#include <memory>
void train_test_pair::decode_xml_node(rapidxml::xml_node<> & node)
{
	using namespace rapidxml;


	assert (string("train_test_pair") == node.name());

	xml_node<> * train_node = node.first_node("train_set");
	shared_ptr<dataset> p_train = deserialize<dataset>(train_node->first_node());
	train.swap(p_train);

	xml_node<> * test_node = node.first_node("test_set");
	shared_ptr<dataset> p_validation = deserialize<dataset>(test_node->first_node());
	test.swap(p_validation);

	train_validation_pairs.resize(0);
	xml_node<> * cv_pairs_node = node.first_node("train_validation_pairs");
	for (xml_node<> * cv_node = cv_pairs_node->first_node("train_validation_pair");
		cv_node != 0 ; cv_node = cv_node->next_sibling("train_validation_pair"))
	{
		shared_ptr<train_validation_pair> p_cv_pair = deserialize<train_validation_pair>(cv_node);
		train_validation_pairs.push_back(*p_cv_pair);
	}

}

#include <boost/lexical_cast.hpp>
#include <string>
using namespace H5;

void train_test_pair::encode_hdf_node(H5::Group * group) const
{

	// Create a fixed-length string
    StrType vls_type(0, 256); // 0 is a dummy argument
    // Open your group
    // Create dataspace for the attribute
    DataSpace att_space(H5S_SCALAR);
    // Create an attribute for the group
    Attribute type_attribute = group->createAttribute("Type",vls_type, att_space);
    // Write data to the attribute
    type_attribute.write(vls_type, "train_test_pair");

	Group train_set_group = group->createGroup("train_set");
	train->encode_hdf_node(&train_set_group);


	Group test_set_group = group->createGroup("test_set");
	test->encode_hdf_node(&test_set_group);


	Group tv_pairs_group = group->createGroup("train_validation_pairs");
	IntType int_type(PredType::NATIVE_INT);
	Attribute tvpairnum_attribute = tv_pairs_group.createAttribute("TVPairNum",int_type, att_space);
    // Write data to the attribute
	int tvpairnum = train_validation_pairs.size();
	tvpairnum_attribute.write(int_type, (void *) &tvpairnum);

	for(int i = 0;i < tvpairnum;i++)
	{
		Group cur_tv_group = tv_pairs_group.createGroup(boost::lexical_cast<std::string>(i));
		train_validation_pairs[i].encode_hdf_node(&cur_tv_group);
	}

}

void train_test_pair::decode_hdf_node(const H5::Group * obj) 
{
	Attribute attr = obj->openAttribute("Type");

	string type_str;
	attr.read(attr.getStrType(),type_str);

	if (type_str != "train_test_pair")
		throw runtime_error("Bad HDF Format"); 

	Group train_group = obj->openGroup("train_set");

	train = deserialize<dataset>(&train_group);


	Group validation_group =  obj->openGroup("test_set");

	test = deserialize<dataset>(&validation_group);


	Group  tv_pairs_group =obj->openGroup( "train_validation_pairs");



	Attribute tvpairnum_attribute = tv_pairs_group.openAttribute("TVPairNum");
    // Write data to the attribute
	int tvpairnum ;
	tvpairnum_attribute.read(tvpairnum_attribute.getIntType(), (void *) &tvpairnum);

	train_validation_pairs.resize(tvpairnum);
	
	for(int i = 0;i < tvpairnum;i++)
	{
		Group cur_tvpair_group = tv_pairs_group.openGroup( boost::lexical_cast<std::string>(i));
		train_validation_pairs[i].decode_hdf_node(&cur_tvpair_group);

	}


}