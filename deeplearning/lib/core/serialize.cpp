#include <liblearning/core/serialize.h>

#include <fstream>
#include <rapidxml/rapidxml_print.hpp>


namespace core
{

	using namespace H5;

	void direct_xml_file_serializable::load_xml(const std::string & filename)
	{
		std::ifstream ifs(filename.c_str());

		std::string content;
		std::getline(ifs,content,(char)EOF);

		ifs.close();

		using namespace rapidxml;
		xml_document<> doc;    
		doc.parse<0>(const_cast<char*>(content.c_str())); 

		std::string  class_name = typeid(*this).name();
		if (! boost::starts_with(class_name,"class "))
			throw class_name + "cannot be deseralized.";

		std::string name = class_name.substr(6);

		xml_node<> * node = doc.first_node(name.c_str());

		this->decode_xml_node(*node);

	}


	void direct_xml_file_serializable::save_xml(const std::string & filename) const
	{
		std::ofstream ofs(filename.c_str());

		using namespace rapidxml;
		xml_document<> doc;    

		doc.append_node(this->encode_xml_node(doc));

		ofs << doc;

		ofs.close();
	}

	boost::mutex hdf_mutex;


	void direct_hdf_file_serializable::load_hdf(const std::string & filename)
	{
		{
			boost::mutex::scoped_lock lock(hdf_mutex);
		
			H5File file( filename, H5F_ACC_RDONLY );

		
			Group root = file.openGroup("/");

		
			this->decode_hdf_node(& root);

			file.close();
		}

	}


	void direct_hdf_file_serializable::save_hdf(const std::string & filename) const
	{
		{
			boost::mutex::scoped_lock lock(hdf_mutex);
			H5File file( filename, H5F_ACC_TRUNC );

			Group root =  file.openGroup("/");

			encode_hdf_node(&root);

			file.close();
		}


	}



	template <>  H5::DataType get_hdf_numeric_datatype<double> ()
	{
		return H5::PredType::NATIVE_DOUBLE;
	}

	template <> H5::DataType get_hdf_numeric_datatype<float> ()
	{
		return H5::PredType::NATIVE_FLOAT;
	}
}