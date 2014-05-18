#include <liblearning/experiment/experiment_datasets.h>

#include <liblearning/core/dataset_group.h>

using namespace experiment;

experiment_datasets::experiment_datasets(void)
{
}


experiment_datasets::~experiment_datasets(void)
{
}


void experiment_datasets::make_folder_based_train_test_pairs(const dataset & data,const dataset_splitter & splitter)
{
	dataset_group group = splitter.split(data);

	train_test_pairs.erase(train_test_pairs.begin(),train_test_pairs.end());

	for (int i = 0;i<group.get_dataset_num();i++)
	{
		shared_ptr<dataset> p_test_set = group.get_dataset(i);

		shared_ptr<dataset> p_train_set;

		for (int j = 0;j < group.get_dataset_num();j++)
		{
			if (j  != i)
			{
				if (!p_train_set)
					p_train_set = group.get_dataset(j)->clone();
				else
					p_train_set->append(*group.get_dataset(j));
			}
		}

		train_test_pairs.push_back(train_test_pair(p_train_set,p_test_set));
	}
}


void experiment_datasets::make_random_splitted_train_test_pairs(const dataset & data,const dataset_splitter & splitter, int pair_num)
{
	train_test_pairs.erase(train_test_pairs.begin(),train_test_pairs.end());

	for(int i = 0;i<pair_num; i++)
	{
		dataset_group group = splitter.split(data);
		if (group.get_dataset_num() != 2)
			throw std::runtime_error("The splitter do not split data set into two parts!");

		shared_ptr<dataset> p_train_set = group.get_dataset(0);

		shared_ptr<dataset> p_test_set = group.get_dataset(1);

		train_test_pairs.push_back(train_test_pair(p_train_set,p_test_set));

	}

}

void experiment_datasets::set_train_test_pairs(const dataset & train, const dataset & test, int pair_num)
{
	shared_ptr<dataset> p_test_set(test.clone());

	shared_ptr<dataset> p_train_set(train.clone());

	train_test_pairs.erase(train_test_pairs.begin(),train_test_pairs.end());

	for (int i = 0; i < pair_num; i++)
	{
		train_test_pairs.push_back(train_test_pair(p_train_set,p_test_set));
	}

}

void experiment_datasets::prepare_cross_validation(const dataset_splitter & splitter)
{
	
	for (int i = 0;i<train_test_pairs.size();i++)
	{
		train_test_pairs[i].make_cross_validation_pairs(splitter);

	}
}

void experiment_datasets::random_split_train_validation(const dataset_splitter & splitter)
{
	
	for (int i = 0;i<train_test_pairs.size();i++)
	{
		train_test_pairs[i].make_random_train_validation_pairs(splitter);

	}
}

int experiment_datasets::get_train_test_pair_num() const
{
	return train_test_pairs.size();
}

const train_test_pair & experiment_datasets::get_train_test_pair(int i ) const
{
	return train_test_pairs[i];
}


rapidxml::xml_node<> * experiment_datasets::encode_xml_node(rapidxml::xml_document<> & doc) const
{
	using namespace rapidxml;


	char * experimental_datasets_name = doc.allocate_string("experiment_datasets"); 
	xml_node<> * experimental_datasets_node = doc.allocate_node(node_element, experimental_datasets_name);


	char * train_test_pairs_name = doc.allocate_string("train_test_pairs"); 
	xml_node<> * train_test_pairs_node = doc.allocate_node(node_element, train_test_pairs_name);

	for(int i = 0;i < train_test_pairs.size();i++)
	{
		train_test_pairs_node->append_node(train_test_pairs[i].encode_xml_node(doc));
	}

	experimental_datasets_node->append_node(train_test_pairs_node);

	return experimental_datasets_node;
}

void experiment_datasets::decode_xml_node(rapidxml::xml_node<> & node)
{
	using namespace rapidxml;


	assert (string("experiment_datasets") == node.name());

	train_test_pairs.resize(0);
	xml_node<> * train_test_pairs_node = node.first_node("train_test_pairs");
	for (xml_node<> * train_test_pair_node = train_test_pairs_node->first_node("train_test_pair");
		train_test_pair_node != 0 ; train_test_pair_node = train_test_pair_node->next_sibling("train_test_pair"))
	{
		shared_ptr<train_test_pair> p_train_test_pair = deserialize<train_test_pair>(train_test_pair_node);
		train_test_pairs.push_back(*p_train_test_pair);
	}

}


using namespace H5;

void experiment_datasets::encode_hdf_node(H5::Group * group) const
{

	// Create a fixed-length string
    StrType vls_type(0, 256); // 0 is a dummy argument
    // Open your group
    // Create dataspace for the attribute
    DataSpace att_space(H5S_SCALAR);
    // Create an attribute for the group
    Attribute type_attribute = group->createAttribute("Type",vls_type, att_space);
    // Write data to the attribute
    type_attribute.write(vls_type, "experiment_datasets");


	Group  tt_pairs_group = group->createGroup("train_test_pairs");
	IntType int_type(PredType::NATIVE_INT);
	DataSpace att_space2(H5S_SCALAR);
	Attribute ttpairnum_attribute = tt_pairs_group.createAttribute("TTPairNum",int_type, att_space2);
    // Write data to the attribute
	int ttpairnum = train_test_pairs.size();
	ttpairnum_attribute.write(int_type, (void *) &ttpairnum);

	for(int i = 0;i < ttpairnum;i++)
	{
		Group cur_tt_group = tt_pairs_group.createGroup(boost::lexical_cast<std::string>(i));
		train_test_pairs[i].encode_hdf_node(&cur_tt_group);

	}


}

void experiment_datasets::decode_hdf_node(const H5::Group * obj) 
{
	Attribute attr = obj->openAttribute("Type");

	string type_str;
	attr.read(attr.getStrType(),type_str);

	if (type_str != "experiment_datasets")
		throw runtime_error("Bad HDF Format"); 

	Group  tt_pairs_group = obj->openGroup( "train_test_pairs");


	Attribute ttpairnum_attribute = tt_pairs_group.openAttribute("TTPairNum");
    // Write data to the attribute
	int ttpairnum ;
	ttpairnum_attribute.read(ttpairnum_attribute.getIntType(), (void *) &ttpairnum);

	train_test_pairs.resize(ttpairnum);
	
	for(int i = 0;i < ttpairnum;i++)
	{
		Group cur_ttpair_group = tt_pairs_group.openGroup( boost::lexical_cast<std::string>(i));
		train_test_pairs[i].decode_hdf_node(&cur_ttpair_group);

	}

}