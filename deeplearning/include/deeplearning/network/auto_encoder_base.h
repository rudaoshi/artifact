#ifndef AUTO_ENCODER_BASE_H
#define AUTO_ENCODER_BASE_H



#include <liblearning/core/dataset.h>
#include <liblearning/core/serialize.h>
#include "neuron_type.h"

#include "layerwise_initializer.h"
#include "network_objective.h"

#include <Eigen/Core>
#include <vector>
#include <memory>
using namespace std;



using namespace Eigen;


namespace deep
{
	class auto_encoder_base : public xml_serializable
	{

	protected:

		vector<int> structure;
		vector<neuron_type> neuron_types;

		// The weight and bias of neurons
		VectorType Wb;
		vector<Map<MatrixType>* > W;
		vector<Map<VectorType>* > b;


		//  the diffence of objective to the weight and bias Wb
		VectorType dWb;

		vector<Map<MatrixType>* > dW;
		vector<Map<VectorType>* > db;

		// Windex[i] is the start of the weights of i-th layer (Input layer is not counted).
		vector<int> Windex;
		// bindex[i] is the start of the bias of i-th layer (Input layer is not counted).
		vector<int> bindex;

		// the initialize error of each layers (Input layer is not counted).
		vector<NumericType> init_layered_error;

		// the input of each layer. (The input layer is not counted)
		// the last element is the out put.
		vector<MatrixType> layered_input;

		// total num of layers of the network. (Input layer is not counted).
		int num_layers;

		//  the position of the encoder layers at all num_layers layers.(Input layer is not counted).
		int coder_layer_id;




	public:

		// compute the delta from the error difference to the output of the 'layer'-th output
		MatrixType error_diff_to_delta(const MatrixType & error_diff, int layer);

		void backprop_output_to_encoder(MatrixType & output_delta);

		void backprop_encoder_to_input(MatrixType &  delta);

		const VectorType& get_Wb() const;
		const VectorType& get_dWb() const;

		MatrixType get_W(int i) const;
		VectorType get_b(int i) const;

		int get_param_num() const;

		void set_Wb(const VectorType& Wb_);

		int get_layer_num();

		int get_output_layer_id();
		int get_coder_layer_id();

		const MatrixType & get_layered_input(int id);

		const MatrixType & get_layered_output(int id);

		void zero_dWb();

	protected:

		auto_encoder_base(const vector<int>& structure,  const vector<neuron_type>& neuron_type);

		auto_encoder_base(const auto_encoder_base & net_);
		auto_encoder_base();

	public:
		virtual ~auto_encoder_base();

		void init(layerwise_initializer & initializer, const dataset & data);

		void init_stacked_rbm(const dataset& data, int num_iter);

		void init_stacked_auto_encoder(const dataset& data, network_objective & trainer, int num_iter);

		void init_random();

		MatrixType encode(const MatrixType& sample) ;

		shared_ptr<dataset> encode(const  dataset & X) ;

		MatrixType decode(const MatrixType & feature) ;

		shared_ptr<dataset> decode(const  dataset & X) ;

		NumericType finetune( const dataset & X, network_objective & obj, int max_iter);

		NumericType finetune_until_converge( const dataset & X, network_objective & obj, int step_iter_num);

		virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

		virtual void decode_xml_node(rapidxml::xml_node<> & node);

				// HDF5 serialization

		virtual void encode_hdf_node(H5::Group * group) const;

		virtual void decode_hdf_node(const H5::Group * obj) ;
	};

}

#endif
