/*
 * deep_auto_encoder.h
 *
 *  Created on: 2010-6-3
 *      Author: sun
 */

#ifndef DEEP_AUTO_ENCODER_H_
#define DEEP_AUTO_ENCODER_H_

#include <liblearning/core/dataset.h>
#include <liblearning/core/serialize.h>
#include <liblearning/deep/neuron_type.h>

#include <liblearning/deep/layerwise_initializer.h>
#include <liblearning/deep/network_objective.h>

#include <liblearning/core/feature_extractor.h>

#include <liblearning/core/maker.h>
#include <liblearning/core/machine.h>
#include <liblearning/core/evaluator_dep.h>

#include <liblearning/optimization/optimizer.h>

#include <vector>
#include <memory>
using namespace std;



namespace deep
{

	struct deep_ae_param
	{
		const vector<int> * structure;
		const vector<neuron_type> * neuron_types;

		shared_ptr<layerwise_initializer> initializer;

		shared_ptr<network_objective> objective;

		shared_ptr<optimization::optimizer> finetune_optimizer;

	};

//	using Eigen::Map;

	class POCO_EXPORT deep_auto_encoder : public feature_extractor, public core::direct_hdf_file_serializable
	{


	private:

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


		shared_ptr<optimization::optimizer> network_optimizer;

		boost::signals2::signal<void (const deep_auto_encoder &)> network_updated;

		boost::signals2::signal<void (const deep_auto_encoder &)> network_trained;


		int batch_size;

		int iter_per_batch;

		int pretrain_iter_num;

		int finetune_iter_num;

		bool mini_batch_opt_finished;

		double best_perf;

		string best_machine_file_path;

		int finetune_running_iter;

	public:

		boost::signals2::signal<void (const deep_auto_encoder &)> networkUpdated;

		boost::signals2::signal<void (const deep_auto_encoder &)> networkTrained;

	public:


		int get_finetune_running_iter();

		void set_batch_setting(int batch_size, int iter_per_batch);

		void set_pretrain_iter_num(int pretrain_iter_num);

		void set_finetune_iter_num(int finetune_iter_num);

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

		void set_optimizer(const shared_ptr<optimization::optimizer> & optimizer_);

		const shared_ptr<optimization::optimizer> & get_optimizer();

		int get_layer_num();

		const vector<int> & get_structure();

		const vector<neuron_type> & get_neuron_types();

		neuron_type get_neuron_type_of_layer(int i);


		int get_output_layer_id();
		int get_coder_layer_id();

		int get_code_layer_dim();
		int get_output_layer_dim();

		const MatrixType & get_layered_input(int id);

		const MatrixType & get_layered_output(int id);

		void zero_dWb();

		MatrixType lastLayerActivation;

		const MatrixType & get_last_layer_activation();


		NumericType finetune_one_batch(const shared_ptr<dataset> & X, network_objective & obj);

	public:

		deep_auto_encoder(const vector<int>& structure,  const vector<neuron_type>& neuron_type);

		deep_auto_encoder(const deep_auto_encoder & net_);
		deep_auto_encoder();
		virtual ~deep_auto_encoder();

// #pragma region Implementing Interface For Feature Extractor
//
// 		virtual void make(const deep_ae_param & param);
//
 		virtual void train(const shared_ptr<dataset> & data);
//
// 		virtual shared_ptr<dataset> extract_feature(const shared_ptr<dataset> & data);
// #pragma endregion

		void init(layerwise_initializer & initializer, const dataset & data);

		void init_stacked_rbm(const dataset& data, int num_iter);

		void init_stacked_auto_encoder(const dataset& data, int rbmiter,network_objective & trainer, const shared_ptr<optimization::optimizer> & optimizer_);

		void init_random();

		MatrixType encode(const MatrixType& sample) ;

		MatrixType decode(const MatrixType & feature) ;

		shared_ptr<dataset> encode(const  dataset & X) ;

		shared_ptr<dataset> decode(const  dataset & X) ;

		NumericType finetune( const shared_ptr<dataset> & X, network_objective & obj);

		void progress_notified(int running_iter);



//		NumericType finetune_until_converge( const dataset & X, network_objective & obj, int step_iter_num);

		//virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

		//virtual void decode_xml_node(rapidxml::xml_node<> & node);

				// HDF5 serialization

		virtual void encode_hdf_node(H5::Group * group) const;

		virtual void decode_hdf_node(const H5::Group * obj) ;
	};
}

CAMP_TYPE(deep::deep_auto_encoder);
#endif /* DEEP_AUTO_ENCODER_H_ */
