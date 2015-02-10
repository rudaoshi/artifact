#include <liblearning/experiment/train_validation_pair.h>

using namespace experiment;

train_validation_pair::train_validation_pair()
{
}

train_validation_pair::train_validation_pair(const shared_ptr<dataset>& train_, const shared_ptr<dataset>& valid_):train(train_),validation(valid_)
{
}
train_validation_pair::~train_validation_pair(void)
{
}

const shared_ptr<dataset> & train_validation_pair::get_train_dataset()  const
{
	return train;
}

const shared_ptr<dataset> & train_validation_pair::get_validation_dataset()  const
{
	return validation;
}

rapidxml::xml_node<> * train_validation_pair::encode_xml_node(rapidxml::xml_document<> & doc) const
{
	using namespace rapidxml;


	char * train_validation_pair_name = doc.allocate_string("train_validation_pair"); 
	xml_node<> * train_validation_pair_node = doc.allocate_node(node_element, train_validation_pair_name);

	char * train_name = doc.allocate_string("train_set"); 
	xml_node<> * train_node = doc.allocate_node(node_element, train_name);
	train_node->append_node(train->encode_xml_node(doc));

	char * validation_name = doc.allocate_string("validation_set"); 
	xml_node<> * validation_node = doc.allocate_node(node_element, validation_name);
	validation_node->append_node(validation->encode_xml_node(doc));

	train_validation_pair_node->append_node(train_node);
	train_validation_pair_node->append_node(validation_node);

	return train_validation_pair_node;
}

#include <memory>
void train_validation_pair::decode_xml_node(rapidxml::xml_node<> & node)
{
	using namespace rapidxml;


	assert (string("train_validation_pair") == node.name());

	xml_node<> * train_node = node.first_node("train_set");
	shared_ptr<dataset> p_train = deserialize<dataset>(train_node->first_node());
	train.swap(p_train);

	xml_node<> * validation_node = node.first_node("validation_set");
	shared_ptr<dataset> p_validation = deserialize<dataset>(validation_node->first_node());
	validation.swap(p_validation);

}

using namespace H5;

void train_validation_pair::encode_hdf_node(H5::Group * group) const
{

	// Create a fixed-length string
    StrType vls_type(0, 256); // 0 is a dummy argument
    // Open your group
    // Create dataspace for the attribute
    DataSpace att_space(H5S_SCALAR);
    // Create an attribute for the group
    Attribute type_attribute = group->createAttribute("Type",vls_type, att_space);
    // Write data to the attribute
    type_attribute.write(vls_type, "train_validation_pair");

	Group train_set_group = group->createGroup("train_set");
	train->encode_hdf_node(&train_set_group);


	Group validation_set_group = group->createGroup("validation_set");
	validation->encode_hdf_node(&validation_set_group);




}

void train_validation_pair::decode_hdf_node(const H5::Group * obj) 
{
	Attribute attr = obj->openAttribute("Type");
	string type_str;
	attr.read(attr.getStrType(),type_str);

	if (type_str != "train_validation_pair")
		throw runtime_error("Bad HDF Format"); 

	Group train_group = obj->openGroup("train_set");

	train = deserialize<dataset>(&train_group);


	Group validation_group = obj->openGroup("validation_set");

	validation = deserialize<dataset>(&validation_group);
	
}