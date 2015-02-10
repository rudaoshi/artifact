#include <liblearning/core/supervised_dataset.h>

#include <liblearning/core/dataset_group.h>
#include <boost/foreach.hpp>

#include <algorithm>
#include <rapidxml/rapidxml_print.hpp>
using namespace core;

supervised_dataset::supervised_dataset(){}

supervised_dataset::supervised_dataset(const MatrixType & data, const vector<int> & label_):dataset(data),label(label_)
{
	calculate_supervised_info();
}

supervised_dataset::supervised_dataset(const supervised_dataset & data_set):dataset(data_set),label(data_set.label)
{
	calculate_supervised_info();
}

supervised_dataset::supervised_dataset(const supervised_dataset & parent_, const vector<int> & index_):dataset(parent_,index_)
{
	label.resize(index.size());
	for (int j = 0; j<index.size(); j++)
	{
		label[j] = parent_.get_label()[index[j]];
	}

	calculate_supervised_info();
}

supervised_dataset::~supervised_dataset(void)
{
}

void supervised_dataset::calculate_supervised_info()
{
	vector<int> temp_class_id = label;
	std::sort(temp_class_id.begin(),temp_class_id.end());
	vector<int>::iterator  end = std::unique(temp_class_id.begin(), temp_class_id.end());

	class_id.resize(end-temp_class_id.begin());
	std::copy(temp_class_id.begin(),end,class_id.begin());

	class_elem_num.resize(class_id.size());

	for (int i = 0;i<class_elem_num.size();i++)
	{
		int elem_num = 0;
		for (vector<int>::iterator iter = label.begin();iter != label.end();iter++)
		{
			if (*iter == class_id[i]) elem_num ++ ;
		}
		class_elem_num[i] = elem_num;

	}
}

const vector<int> & supervised_dataset::get_label() const
{
	return label;
}

const vector<int> & supervised_dataset::get_class_id() const
{
	return class_id;
}

int supervised_dataset::get_class_num() const
{
	return class_id.size();
}

const vector<int> & supervised_dataset::get_class_elem_num() const
{
	return class_elem_num;
}

void supervised_dataset::append(const dataset & data_set)
{
	const supervised_dataset & s_data_set = dynamic_cast<const supervised_dataset &>(data_set);

	dataset::append(data_set);

	label.insert(label.end(),s_data_set.get_label().begin(),s_data_set.get_label().end());
}

void supervised_dataset::copy(const dataset & data_set)
{
	const supervised_dataset & s_data_set = dynamic_cast<const supervised_dataset &>(data_set);

	dataset::copy(data_set);

	label = s_data_set.get_label();
}

shared_ptr<dataset> supervised_dataset::clone() const
{
	const dataset * pthis = this;
	return shared_ptr<dataset>(new supervised_dataset(*this));
}

shared_ptr<dataset> supervised_dataset::clone_update_data(const MatrixType & data) const
{
	return shared_ptr<dataset>(new supervised_dataset(data, label));
}

shared_ptr<dataset> supervised_dataset::sub_set(const vector<int> & index_) const
{
	return shared_ptr<dataset>(new supervised_dataset(*this, index_));
}


#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>


rapidxml::xml_node<> * supervised_dataset::encode_xml_node(rapidxml::xml_document<> & doc) const
{
	using namespace rapidxml;



	char * dataset_name = doc.allocate_string("supervised_dataset"); 
	xml_node<> * dataset_node = doc.allocate_node(node_element, dataset_name);

	char * dim_name = doc.allocate_string("dim"); 
	char * dim_value = doc.allocate_string(boost::lexical_cast<string >(get_dim()).c_str()); 
	xml_node<> * dim_node = doc.allocate_node(node_element, dim_name, dim_value);

	char * sample_num_name = doc.allocate_string("sample_num"); 
	char * sample_num_value = doc.allocate_string(boost::lexical_cast<string>(get_sample_num()).c_str()); 
	xml_node<> * sample_num_node = doc.allocate_node(node_element, sample_num_name, sample_num_value);

	std::ostringstream sample_ss;
	sample_ss << (EigenMatrixType)data;
	char * samples_name = doc.allocate_string("samples"); 
	char * samples_value = doc.allocate_string(sample_ss.str().c_str()); 
	xml_node<> * samples_node = doc.allocate_node(node_element, samples_name, samples_value);

	std::ostringstream indexs_ss;
	BOOST_FOREACH(int x,index){indexs_ss << x << ' ';}
//	for_each(index.begin(),index.end(),[&indexs_ss](int n){indexs_ss << n << ' ';});
	char * indexs_name = doc.allocate_string("indexs"); 
	char * indexs_value = doc.allocate_string(indexs_ss.str().c_str()); 
	xml_node<> * indexs_node = doc.allocate_node(node_element, indexs_name, indexs_value);

	std::ostringstream labels_ss;
	BOOST_FOREACH(int x,label){labels_ss << x << ' ';}
//	for_each(label.begin(),label.end(),[&labels_ss](int n){labels_ss << n << ' ';});
	char * labels_name = doc.allocate_string("labels"); 
	char * labels_value = doc.allocate_string(labels_ss.str().c_str()); 
	xml_node<> * labels_node = doc.allocate_node(node_element, labels_name, labels_value);

	dataset_node->append_node(dim_node);
	dataset_node->append_node(sample_num_node);
	dataset_node->append_node(samples_node);
	dataset_node->append_node(indexs_node);
	dataset_node->append_node(labels_node);

	return dataset_node;

}


void supervised_dataset::decode_xml_node(rapidxml::xml_node<> & node)
{
	using namespace rapidxml;


	dataset::decode_xml_node(node);
	
	xml_node<> * labels_node =  node.first_node("labels");
	string labels_str = labels_node->value();

	std::istringstream label_iss(labels_str,istringstream::in);

	//typedef vector< string > split_vector_type;
	//split_vector_type label_v;
	//boost::split( label_v, labels_str, is_space() );

	//auto new_end = std::remove_if(label_v.begin(),label_v.end(),[](const string& str){ return str.empty(); });

	//int sample_num = data.cols();
	//if (new_end - label_v.begin() != sample_num)
	//	throw "Bad data file: the label num does not equal to sample_num!";

	int sample_num = data.cols();
	label.resize(sample_num);

	for (int j = 0; j< sample_num;j++)
	{
		label_iss >> label[j];// = lexical_cast<int>(label_v[j]);
	}

	calculate_supervised_info();
}



using namespace H5;

void supervised_dataset::encode_hdf_node(H5::Group * group) const
{

	// Create a fixed-length string
    StrType vls_type(0, 256); // 0 is a dummy argument
    // Open your group
    // Create dataspace for the attribute
    DataSpace att_space(H5S_SCALAR);
    // Create an attribute for the group
    Attribute type_attribute = group->createAttribute("Type",vls_type, att_space);
    // Write data to the attribute
    type_attribute.write(vls_type, "supervised_dataset");

	hsize_t data_dims[2] = { this->get_sample_num(), this->get_dim()};
	DataSpace data_space( 2, data_dims );


	DataSet data_block = group->createDataSet("Data", get_hdf_numeric_datatype<NumericType>(),data_space);

	const NumericType * data_ptr;

#ifdef USE_GPU
	EigenMatrixType temp_data = (EigenMatrixType)this->get_data();
	data_ptr = temp_data.data();
#else
	data_ptr = this->get_data().data();
#endif

	data_block.write((void *)data_ptr, get_hdf_numeric_datatype<NumericType>(), data_space, data_space );

	if (this->get_index().size() != 0)
	{
		hsize_t index_dims[] = {this->get_index().size()};  

		DataSpace index_space(1, index_dims );

		DataSet index_block = group->createDataSet("Index", PredType::NATIVE_INT, index_space);

		index_block.write((void *)&this->get_index()[0], PredType::NATIVE_INT, index_space, index_space );

	}

	hsize_t label_dims[] = {this->get_label().size()};  

	DataSpace label_space(1, label_dims );

	DataSet label_block  = group->createDataSet("Label", PredType::NATIVE_INT,  label_space);

	label_block.write((void *)&this->get_label()[0], PredType::NATIVE_INT, label_space, label_space );



}

void supervised_dataset::decode_hdf_node(const H5::Group * obj) 
{
	Attribute attr = obj->openAttribute("Type");

	string type_str;
	attr.read(attr.getStrType(),type_str);

	if (type_str != "supervised_dataset")
		throw runtime_error("Bad HDF Format"); 

	DataSet data_block = obj->openDataSet("Data");

	DataSpace data_space = data_block.getSpace();

	int rank = data_space.getSimpleExtentNdims();

	if (rank != 2)
		throw runtime_error("the rank of the data set is not equal to 2");

	hsize_t dims_out[2];
	int ndims = data_space.getSimpleExtentDims( dims_out, NULL);

	int sample_num = dims_out[0];
	int dim = dims_out[1];

	data.resize(dim,sample_num);

#ifdef USE_GPU
	EigenMatrixType temp_data(dim,sample_num);
	data_block.read( temp_data.data(), get_hdf_numeric_datatype<NumericType>(), data_space, data_space );
	data = temp_data;
#else
	data_block.read( data.data(), get_hdf_numeric_datatype<NumericType>(), data_space, data_space );
#endif

	try
	{

		DataSet index_block = obj->openDataSet("Index");

		DataSpace index_space = index_block.getSpace();

		rank = index_space.getSimpleExtentNdims();

		if (rank != 1)
			throw runtime_error("the rank of the index is not equal to 1");

		hsize_t rank_dims_out[1];
		ndims = index_space.getSimpleExtentDims( rank_dims_out, NULL);

		int index_num = rank_dims_out[0];

		index.resize(index_num);

		index_block.read( &index[0], PredType::NATIVE_INT, index_space, index_space );
	}
	catch (...)// No index 
	{

	}

	DataSet label_block = obj->openDataSet("Label");

	DataSpace label_space = label_block.getSpace();

	rank = label_space.getSimpleExtentNdims();

	if (rank != 1)
		throw runtime_error("the rank of the label is not equal to 1");

	hsize_t label_dims_out[1];
	ndims = label_space.getSimpleExtentDims( label_dims_out, NULL);

	int label_num = label_dims_out[0];

	label.resize(label_num);

	label_block.read(&label[0], PredType::NATIVE_INT, label_space, label_space );

	calculate_supervised_info();
}

