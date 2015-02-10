#ifndef SERIALIZE_H_
#define SERIALIZE_H_

#include <camp/camptype.hpp>
#include <camp/class.hpp>


#include <rapidxml/rapidxml.hpp>

#include <boost/algorithm/string.hpp>

#include <fstream>

#include <memory>

#include <string>
#include <H5Cpp.h>


#include <iostream>
#include <stdexcept>
#include <liblearning/core/config.h>

#include <boost/thread.hpp>

using namespace H5;
using namespace std;
namespace core
{





class xml_serializable
{
public:

	virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const = 0;

	virtual void decode_xml_node(rapidxml::xml_node<> & node) = 0;
};

class hdf_serializable
{
public:

	virtual void encode_hdf_node(H5::Group * obj) const = 0;

	virtual void decode_hdf_node(const H5::Group * obj) = 0;
};

class direct_hdf_file_serializable: public hdf_serializable
{
public:
	virtual void load_hdf(const std::string & filename) ;

	virtual void save_hdf(const std::string & filename) const;
};


class direct_xml_file_serializable : public xml_serializable
{
public:

	virtual void load_xml(const std::string & filename);

	virtual void save_xml(const std::string & filename) const ;
};


template <typename T> H5::DataType get_hdf_numeric_datatype();

template <> H5::DataType get_hdf_numeric_datatype<double> ();

template <> H5::DataType get_hdf_numeric_datatype<float> ();


template <typename ST>
shared_ptr<ST> deserialize(const H5::Group * group)
{
	int numAttr = group->getNumAttrs();

	if (numAttr <1)
		throw runtime_error("Cannot determine the type of the object");

	Attribute attr = group->openAttribute("Type");

	string type_str;
	attr.read(attr.getStrType(),type_str);

	const camp::Class& metaclass = camp::classByName(type_str);

	shared_ptr<ST> obj(metaclass.construct<ST>());

	obj->decode_hdf_node(group);

	return obj;
 
}


template <typename ST>
shared_ptr<ST> deserialize(rapidxml::xml_node<> * node)
{
	const camp::Class& metaclass = camp::classByName(node->name());

	shared_ptr<ST> obj(metaclass.construct<ST>());

	obj->decode_xml_node(* node);

	return obj;
 
}


template <typename ST>
shared_ptr<ST> deserialize_from_xml_file(const std::string & filename)
{
	std::ifstream ifs(filename.c_str());

	if (!ifs)
		throw runtime_error("Invalid file!");

	std::string content;
	std::getline(ifs,content,(char)EOF);

	ifs.close();

	using namespace rapidxml;
	xml_document<> doc;    
	doc.parse<0>(const_cast<char*>(content.c_str())); 

	xml_node<> * node = doc.first_node();

	return deserialize<ST>(node);
 
}

extern boost::mutex hdf_mutex;

template <typename ST>
shared_ptr<ST> deserialize_from_hdf_file(const std::string & filename)
{
	{
		boost::mutex::scoped_lock lock(hdf_mutex);

		H5File file( filename, H5F_ACC_RDONLY );

		Group root_group = file.openGroup("/");

		int numAttr = root_group.getNumAttrs();

		if (numAttr <1)
			throw runtime_error("Cannot determine the type of the object");

		shared_ptr<ST> obj = deserialize<ST>(&root_group);

		return obj;
	}

}


}
#endif
