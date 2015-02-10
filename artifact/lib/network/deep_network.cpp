/*
* deep_auto_encoder.cpp
*
*  Created on: 2010-6-3
*      Author: sun
*/

#include <artifact/network/deep_network.h>
#include <cassert>

#include <artifact/util/math_util.h>
#include <artifact/util/matrix_util.h>
#include <artifact/network/restricted_boltzmann_machine.h>

#include <algorithm>
#include <artifact/network/neuron_layer_operation.h>

#include <artifact/network/network_optimobj_adapter.h>
#include <artifact/optimization/conjugate_gradient_optimizer.h>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/foreach.hpp>
#include <artifact/util/algo_util.h>

#include <artifact/core/data_splitter.h>
#include <artifact/core/supervised_dataset.h>
#include <artifact/core/platform.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <boost/filesystem.hpp>


using namespace std;

using namespace artifact::network;

    
vector<MatrixType> deep_network::feed_forward(const MatrixType &input)
{
    vector<MatrixType> result;
    MatrixType cur_layer_input = input;
    for (int i=0; i<layers.size(); i++)
    {
        result.push_back(layers[i].predict(cur_layer_input));
    }

    return result;
}
    
vector<VectorType> deep_network::back_propagate(const MatrixType & input,
        const VectorType & y,
        const vector<MatrixType> & laywise_output)
{
    MatrixType delta = MaxtrixType::Zeros(laywise_output[layers.size() - 1].rows(),
            laywise_output[layers.size() - 1].cols());
    vector<VectorType> gradients(layers.size());

    for (int i = layers.size() - 1; i >= 0; i --)
    {
        MatrixType * cur_input = 0;

        if (i > 0)
            cur_input = &laywise_output[i-1];
        else
            cur_input = &input;

        MatrixType * cur_ouput = *laywise_output[i];

        mlp_layer * layer = &this->layers[i];
        if (layer->loss_func)
        {
            delta += layer->loss_func(* cur_output);
        }

        delta = layer->.backprop_delta(
                delta, *input, *output);

        gradients[i] = layer->compute_param_gradient(
                delta, *input, *output);
    }
}


	deep_network::deep_network()
	{
	}

	deep_network::deep_network(const deep_network & net_):parameter_host(net_)
	{
// 		structure = net_.structure;
// 		neuron_types = net_.neuron_types;

		Wb = net_.Wb;

		Windex = net_.Windex;
		bindex = net_.bindex;

// 		num_layers = net_.num_layers;

		//  the position of the encoder layers at all num_layers layers.(Input layer is not counted).
// 		coder_layer_id = net_.coder_layer_id;

		//  the diffence of objective to the weight and bias Wb
		dWb.resize(Wb.size());

		layered_input.resize(num_layers+1);
		init_layered_error.resize(num_layers/2);

		W.resize(structure.size()-1);
		b.resize(structure.size()-1);

		dW.resize(structure.size()-1);
		db.resize(structure.size()-1);
		for (int level = 0; level < num_layers; level ++)
		{
			W[level] = new Map<MatrixType>(Wb.data()+ Windex[level],structure[level + 1],structure[level]);
			b[level] = new Map<VectorType>(Wb.data()+ bindex[level],structure[level + 1]);

			dW[level] = new Map<MatrixType>(dWb.data()+ Windex[level],structure[level + 1],structure[level]);
			db[level] = new Map<VectorType>(dWb.data()+ bindex[level],structure[level + 1]);

		}

		batch_size = 0;
		iter_per_batch = 0;

		best_perf = 0;

		best_machine_file_path = "";

		finetune_iter_num = 0;

	}
	deep_auto_encoder::~deep_auto_encoder()
	{
// 		for (int level = 0; level < num_layers; level ++)
// 		{
// 			delete W[level] ;
// 			delete b[level] ;
//
// 			delete dW[level] ;
// 			delete db[level] ;
//
// 		}

	}


#pragma region Implementing Interface For Feature Extractor

void deep_network::make(const shared_ptr<deep_ae_param> & param_)
{
	param =  param_;

	assert(param->structure->size() == param->neuron_types->size()+1);
	num_layers = 2*(param->structure->size()-1);

	coder_layer_id = param->structure->size()-2;

	structure.resize(1+num_layers);
	neuron_types.resize(num_layers);

	std::copy(param->structure->begin(),param->structure->end(),structure.begin());
	std::copy(param->neuron_types->begin(),param->neuron_types->end(),neuron_types.begin());

	std::copy(param->structure->rbegin()+1,param->structure->rend(),structure.begin()+param->structure->size());
	std::fill(neuron_types.begin()+param->neuron_types->size(),neuron_types.end(),logistic );


	layered_input.resize(num_layers+1);

	init_layered_error.resize(num_layers/2);

	Windex.resize(structure.size()-1);
	bindex.resize(structure.size()-1);

	W.resize(structure.size()-1);
	b.resize(structure.size()-1);

	dW.resize(structure.size()-1);
	db.resize(structure.size()-1);

	int  W_ind = 0;

	for (int level = 0; level < num_layers; level ++)
	{

		Windex[level] = W_ind;
		bindex[level] = W_ind + structure[level] * structure[level + 1];

		W_ind = W_ind + structure[level] * structure[level + 1] + structure[level + 1];
	}

	Wb.setZero(W_ind);
	dWb.setZero(W_ind);

	for (int level = 0; level < num_layers; level ++)
	{
		W[level] = new Map<MatrixType>(Wb.data()+ Windex[level],structure[level + 1],structure[level]);
		b[level] = new Map<VectorType>(Wb.data()+ bindex[level],structure[level + 1]);

		dW[level] = new Map<MatrixType>(dWb.data()+ Windex[level],structure[level + 1],structure[level]);
		db[level] = new Map<VectorType>(dWb.data()+ bindex[level],structure[level + 1]);

	}

		batch_size = 0;
	iter_per_batch = 0;

	best_perf = 0;

			best_machine_file_path = "";

		finetune_iter_num = 0;
}

void deep_auto_encoder::train(const shared_ptr<dataset> & data)
{
	this->init(*param->initializer,*data);
	this->set_optimizer(param->finetune_optimizer);

	this->finetune(data,*param->objective);
}

shared_ptr<dataset> deep_auto_encoder::extract_feature(const shared_ptr<dataset> & data)
{
	return this->encode(*data);
}
#pragma endregion


	MatrixType deep_auto_encoder::get_W(int i) const
	{
		return *W[i];
	}
	VectorType deep_auto_encoder::get_b(int i) const
	{
		return *b[i];
	}

	void deep_auto_encoder::init(layerwise_initializer & initializer, const dataset & data)
	{
		shared_ptr<dataset> cur_train_data (data.clone());
		for(int i = 0; i < num_layers/2;i++)
		{
			neuron_type type = neuron_types[i];

			initializer.init(structure[i], structure[i+1], type);

			NumericType cur_error = initializer.train(cur_train_data);
			cur_train_data = initializer.get_output();


			*W[i] = initializer.get_W1();;

			*b[i] =  initializer.get_b1();;

			*W[num_layers-i-1] = initializer.get_W2();

			*b[num_layers-i-1] = initializer.get_b2();
			//				memcpy(Wb.data()+Windex[num_layers-i], tW.data(), structure[i]*structure[i+1]*sizeof(NumericType));
			//				memcpy(Wb.data()+bindex[num_layers-i], curRBM.get_b().data(), structure[i+1]*sizeof(NumericType));

			init_layered_error[i] = cur_error;
		}

	}

	void deep_auto_encoder::init_stacked_rbm(const dataset& data, int num_iter)
	{
		/*			NumericType min_train = data.minCoeff();
		NumericType max_train = data.maxCoeff();
		if ( (min_train < 0) || (max_train > 1) )
		throw "data needs to be scaled between 0 and 1!";
		*/
		shared_ptr<dataset> RBM_data (data.clone());
		for(int i = 0; i < num_layers/2;i++)
		{
			neuron_type type = neuron_types[i];

			restricted_boltzmann_machine curRBM(structure[i], structure[i+1], type);
			curRBM.set_batch_setting(batch_size, iter_per_batch);
			NumericType cur_RBMerror = 0;
			cur_RBMerror = curRBM.train(RBM_data, num_iter);
			RBM_data = curRBM.output(*RBM_data);
			/*}
			else
			{
				cur_RBMerror = curRBM.train_batch(RBM_data,batch_size,iter_per_batch,num_iter);
				RBM_data = curRBM.output_batch(*RBM_data,batch_size);
			}
			*/


			*W[i] = curRBM.get_W();
			*b[i] = curRBM.get_c();

			//				memcpy(Wb.data()+Windex[i], curRBM.get_W().data(), structure[i]*structure[i+1]*sizeof(NumericType));
			//				memcpy(Wb.data()+bindex[i], curRBM.get_c().data(), structure[i+1]*sizeof(NumericType));


			MatrixType tW = curRBM.get_W().transpose();


			*W[num_layers-i-1] = tW;

			*b[num_layers-i-1] = curRBM.get_b();
			//				memcpy(Wb.data()+Windex[num_layers-i], tW.data(), structure[i]*structure[i+1]*sizeof(NumericType));
			//				memcpy(Wb.data()+bindex[num_layers-i], curRBM.get_b().data(), structure[i+1]*sizeof(NumericType));

			init_layered_error[i] = cur_RBMerror;
		}
	}

	void deep_auto_encoder::init_stacked_auto_encoder(const dataset& data, int rbmiter,network_objective & objective,const shared_ptr<optimization::optimizer> & optimizer_)
	{

		/*	NumericType min_train = data.minCoeff();
		NumericType max_train = data.maxCoeff();
		if ( (min_train < 0) || (max_train > 1) )
		throw "data needs to be scaled between 0 and 1!";
		*/

		shared_ptr<dataset> auto_encoder_data (data.clone());
		for(int i = 0; i < num_layers/2;i++)
		{
			std::vector<int> cur_structure(2);
			std::vector<neuron_type> cur_type(1);

			cur_structure[0] = structure[i];
			cur_structure[1] = structure[i+1];
			cur_type[0] = neuron_types[i];

			deep_auto_encoder cur_auto_encoder = deep_auto_encoder(cur_structure, cur_type);
			cur_auto_encoder.set_batch_setting(batch_size, iter_per_batch);
			cur_auto_encoder.init_stacked_rbm(*auto_encoder_data, rbmiter);

			cur_auto_encoder.set_optimizer(optimizer_);

			cur_auto_encoder.finetune(auto_encoder_data, objective);

			auto_encoder_data = cur_auto_encoder.encode(*auto_encoder_data);

			*W[i] = cur_auto_encoder.get_W(0);
			*b[i] = cur_auto_encoder.get_b(0);

			*W[num_layers-i-1] = cur_auto_encoder.get_W(1);
			*b[num_layers-i-1] = cur_auto_encoder.get_b(1);

		}

	}

	void deep_auto_encoder::init_random()
	{
		for (int i = 0;i < num_layers;i++)
		{
			*W[i] = (1/sqrt(NumericType(W[i]->cols()))* (2*rand<MatrixType>(W[i]->rows(),W[i]->cols()).array()-1));
			*b[i] = (1/sqrt(NumericType(b[i]->cols()))* (2*rand<VectorType>(b[i]->size()).array()-1));

		}

	}

	void deep_auto_encoder::set_batch_setting(int batch_size_, int iter_per_batch_)
	{
		batch_size = batch_size_;
		iter_per_batch = iter_per_batch_;
	}

	void deep_auto_encoder::set_pretrain_iter_num(int pretrain_iter_num_)
	{
		pretrain_iter_num = pretrain_iter_num_;
	}

	void deep_auto_encoder::set_finetune_iter_num(int finetune_iter_num_)
	{
		finetune_iter_num = finetune_iter_num_;
	}

	int deep_auto_encoder::get_layer_num()
	{
		return num_layers;
	}

	int deep_auto_encoder::get_output_layer_id()
	{
		return num_layers - 1 ;
	}

	int deep_auto_encoder::get_coder_layer_id()
	{
		return coder_layer_id;
	}

	int deep_auto_encoder::get_code_layer_dim()
	{
		return structure[coder_layer_id+1];
	}
	int deep_auto_encoder::get_output_layer_dim()
	{
		return structure[structure.size()-1];
	}

	const MatrixType & deep_auto_encoder::get_layered_input(int id)
	{
		return layered_input[id];
	}

	const MatrixType & deep_auto_encoder::get_layered_output(int id)
	{
		return layered_input[id+1];
	}

	const vector<int> &  deep_auto_encoder::get_structure()
	{
		return structure;
	}

	const vector<neuron_type> &  deep_auto_encoder::get_neuron_types()
	{
		return neuron_types;
	}

	neuron_type deep_auto_encoder::get_neuron_type_of_layer(int i)
	{
		return neuron_types[i];
	}

	void deep_auto_encoder::zero_dWb()
	{
		dWb.setZero(dWb.size());
	}

	MatrixType deep_auto_encoder::encode(const  MatrixType & sample)
	{



		layered_input[0] = sample;

		for (int level = 0; level <= coder_layer_id; level ++)
		{
			if (neuron_types[level] ==  linear)
			{
				linear_layer_output(layered_input[level+1], *W[level], 'N', layered_input[level], *b[level]);

				//do linear level
				//linear_transform(layered_output[level+1].data(), structure[level+1], structure[level], sample.cols(), 1.0, Wb.data()+Windex[level], 'N', layered_output[level].data(), 1, Wb.data()+bindex[level]);

			}
			else
			{
				logistic_layer_output(layered_input[level+1], *W[level], 'N',layered_input[level] ,  *b[level]);

				// do logistic unit levels
				// logistic_transform(layered_output[level+1].data(), structure[level+1], structure[level], sample.cols(), -1.0, Wb.data()+Windex[level], 'N', layered_output[level].data(), -1, Wb.data()+bindex[level]);

			}
		}

		return layered_input[coder_layer_id+1];
	}


	shared_ptr<dataset> deep_auto_encoder::encode(const  dataset & X)
	{
		if ( batch_size == 0 )
		{
			MatrixType Y_data = encode(X.get_data());
			return X.clone_update_data(Y_data);
		}
		else
		{
			int N = X.get_sample_num();
			int batch_num = (N + batch_size -1)/batch_size;

			int code_layer_dim = get_code_layer_dim();

			MatrixType Y(code_layer_dim,N);
			for (int i = 0;i<batch_num;i++)
			{
				int cur_batch_size = batch_size;
				if (i == batch_num-1)
					cur_batch_size = N - (batch_num-1)*batch_size;

				MatrixType cur_X = X.get_data().block(0,i*batch_size,X.get_dim(),cur_batch_size);
				Y.block(0,i*batch_size,code_layer_dim,cur_batch_size) = encode(cur_X);

			}

			return X.clone_update_data(Y);
		}

	}

	MatrixType deep_auto_encoder::decode(const  MatrixType & feature)
	{

		layered_input[coder_layer_id+1] = feature;

		for (int level = coder_layer_id+1; level < num_layers; level ++)
		{

			if (neuron_types[level] ==  linear)
			{
				linear_layer_output(layered_input[level+1],*W[level], 'N', layered_input[level], *b[level]);

			}
			else
			{
				logistic_layer_output(layered_input[level+1], *W[level], 'N', layered_input[level], *b[level]);
			}


		}

		return layered_input[num_layers];
	}

	shared_ptr<dataset> deep_auto_encoder::decode(const  dataset & X)
	{
		if (batch_size == 0)
		{
			MatrixType Y_data = decode(X.get_data());
			return X.clone_update_data(Y_data);
		}
		else
		{
			int N = X.get_sample_num();
			int batch_num = (N + batch_size -1)/batch_size;

			int output_layer_dim = get_output_layer_dim();
			MatrixType Y(output_layer_dim,N);
			for (int i = 0;i<batch_num;i++)
			{
				int cur_batch_size = batch_size;
				if (i == batch_num-1)
					cur_batch_size = N - (batch_num-1)*batch_size;

				MatrixType cur_X = X.get_data().block(0,i*batch_size,X.get_dim(),cur_batch_size);
				Y.block(0,i*batch_size,output_layer_dim,cur_batch_size) = decode(cur_X);

			}

			return X.clone_update_data(Y);
		}

	}

	//shared_ptr<dataset> deep_auto_encoder::encode(const  dataset & X,int batch_size)
	//{

	//	int N = X.get_sample_num();
	//	int batch_num = (N + batch_size -1)/batch_size;

	//	int code_layer_dim = get_code_layer_dim();

	//	MatrixType Y(code_layer_dim,N);
	//	for (int i = 0;i<batch_num;i++)
	//	{
	//		int cur_batch_size = batch_size;
	//		if (i == batch_num-1)
	//			cur_batch_size = N - (batch_num-1)*batch_size;

	//		MatrixType cur_X = X.get_data().block(0,i*batch_size,X.get_dim(),cur_batch_size);
	//		Y.block(0,i*batch_size,code_layer_dim,cur_batch_size) = encode(cur_X);
	//
	//	}

	//	return X.clone_update_data(Y);
	//}

	/*shared_ptr<dataset> deep_auto_encoder::decode_batch(const  dataset & X, int batch_size)
	{

		int N = X.get_sample_num();
		int batch_num = (N + batch_size -1)/batch_size;

		int output_layer_dim = get_output_layer_dim();
		MatrixType Y(output_layer_dim,N);
		for (int i = 0;i<batch_num;i++)
		{
			int cur_batch_size = batch_size;
			if (i == batch_num-1)
				cur_batch_size = N - (batch_num-1)*batch_size;

			MatrixType cur_X = X.get_data().block(0,i*batch_size,X.get_dim(),cur_batch_size);
			Y.block(0,i*batch_size,output_layer_dim,cur_batch_size) = decode(cur_X);

		}

		return X.clone_update_data(Y);
	}
*/



	const VectorType& deep_auto_encoder::get_Wb() const
	{
		return Wb;
	}



	const VectorType & deep_auto_encoder::get_dWb() const
	{
		return dWb;
	}


	int deep_auto_encoder::get_param_num() const
	{
		return Wb.size();
	}


	void deep_auto_encoder::set_Wb(const VectorType& Wb_)
	{
		Wb = Wb_;
	}

	void deep_auto_encoder::set_optimizer(const shared_ptr<optimization::optimizer> & optimizer_)
	{
		network_optimizer = optimizer_;
	}

	const shared_ptr<optimization::optimizer> & deep_auto_encoder::get_optimizer()
	{
		return network_optimizer;
	}


	NumericType deep_auto_encoder::finetune_one_batch(const shared_ptr<dataset> & X,   network_objective& obj)
	{
//		std::cout << "  Setting the Data Set "  << std::endl ;
		obj.set_dataset(X);
//		std::cout << "  DataSet Setted "  << std::endl ;
		if (obj.get_type() == encoder_related)
		{
			for (int i = coder_layer_id+1; i < num_layers; i++)
			{
				W[i]->setZero();
				b[i]->setZero();

			}
		}
		const NumericType EPS=1.0e-8;

		//optimization::conjugate_gradient_optimizer optimizer(max_iter, EPS);

		NumericType obj_val = 0;


		network_optimobj_adapter optim_obj(*this, obj);
//		std::cout << "  Optimization Begin "  << std::endl ;
		tie(obj_val,Wb) = network_optimizer->optimize(optim_obj,Wb);
//		std::cout << "  Optimization Finished "  << std::endl ;

//		std::cout<<obj.get_info();
		return obj_val;
	}

	NumericType deep_auto_encoder::finetune(const shared_ptr<dataset> & X, network_objective & obj)
	{
		NumericType obj_val = 0;
		if (batch_size == 0)
		{
			obj_val =  finetune_one_batch(X,obj);
		}
		else
		{

			int N = X->get_sample_num();

			int batch_num = (N + batch_size -1)/batch_size;



			for (int curr_iter = 0; curr_iter < finetune_iter_num; curr_iter++)
			{
				std::cout << std::endl << "Batch Fintune Iteration " << curr_iter << "  Begins!" << std::endl;
				obj_val = 0;

				if (batch_num == 1)
				{
					mini_batch_opt_finished = false;
					obj_val = finetune_one_batch(X,obj);
					mini_batch_opt_finished = true;
					progress_notified();
				}
				else
				{
					mini_batch_opt_finished = false;
								dataset_group group;
					shared_ptr<dataset_splitter> splitter;
					const supervised_dataset * data = dynamic_cast<const supervised_dataset *>(&(*X));
					if (data == 0)
					{
						splitter.reset(new random_shuffer_dataset_splitter(batch_num));
					}
					else
					{
						splitter.reset(new supervised_random_shuffer_dataset_splitter(batch_num));
					}
					group = splitter->split(*X);

					for (int batch_id = 0; batch_id < group.get_dataset_num(); batch_id++)
					{
						std::cout << "  Begining Finetune Batch " << batch_id << ", Procedure :" ;
	//					std::cout << "  Getting the Batch " << batch_id << std::endl ;
						shared_ptr<dataset> cur_data = group.get_dataset(batch_id);
	//					std::cout << "  Batch " << batch_id << " Has Been Getted" << std::endl ;
						obj_val += finetune_one_batch(cur_data,obj);

						std::cout << std::endl;
					}
					mini_batch_opt_finished = true;
					progress_notified();
					obj_val /= group.get_dataset_num();
				}

				std::cout << std::endl;

				this->save_hdf(boost::lexical_cast<string>(curr_iter) + ".hdf");
			}

		}
		if (best_machine_file_path != "")
		{
			this ->load_hdf(best_machine_file_path);
		}
		return obj_val;
	}

	void deep_auto_encoder::progress_notified()
	{
		if (batch_size != 0 && mini_batch_opt_finished == false)
			return;

		if (!perf_evaluator)
			return;

		vector<double> param;
		double perf = perf_evaluator->evaluate(*this,param);

		if (perf > best_perf)
		{
			best_perf = perf;

			//std::cout << "Machine with higher performance found! Current Best Perf = "<< best_perf << "!";
			//
			//if (param.size() > 0)
			//{
			//	std::cout << "Optimal param:";
			//
			//	for (int i = 0;i < param.size(); i++)
			//	{
			//		std::cout << param[i] << std::endl;
			//	}
			//}

			boost::uuids::uuid tag = boost::uuids::random_generator()();

			string old_machine_file_path = best_machine_file_path;

			best_machine_file_path = boost::lexical_cast<std::string>(tag)+".hdf";

			this->save_hdf(best_machine_file_path);

			//std::cout << "Current best machine has been saved to  "<< best_machine_file_path << std::endl;


			if (old_machine_file_path != "")
				boost::filesystem::remove(boost::filesystem::path(old_machine_file_path));
		}
		else if (perf == best_perf)
		{
			//std::cout << "Performance remains not changed! Current Perf = "<< perf << "! ";
			//
			//if (param.size() > 0)
			//{
			//	std::cout << "Optimal param:";
			//
			//	for (int i = 0;i < param.size(); i++)
			//	{
			//		std::cout << param[i] << std::endl;
			//	}
			//}
		}
		else
		{
			//std::cout << "Performance Dropped! Current Perf = "<< perf << "Current Best Perf = "<< best_perf <<"!";
			//
			//if (param.size() > 0)
			//{
			//	std::cout << "Optimal param:";
			//
			//	for (int i = 0;i < param.size(); i++)
			//	{
			//		std::cout << param[i] << std::endl;
			//	}
			//}
		}


	}

	//NumericType deep_auto_encoder::finetune_until_converge( const dataset & X, network_objective & obj, int step_iter_num)
	//{
	//	obj.set_dataset(X);

	//	NumericType old_obj_val = obj.value(*this);

	//	network_optimobj_adapter optim_obj(*this, obj);

	//	NumericType new_obj_val = 0;

	//	NumericType ftol = 1e-10;

	//	const NumericType EPS=1.0e-18;

	//	do
	//	{

	//		//optimization::conjugate_gradient_optimizer optimizer(step_iter_num, 1e-10);


	//		tie(new_obj_val,Wb) = network_optimizer->optimize(optim_obj,Wb);
	//	}
	//	while(2.0*fabs(old_obj_val-new_obj_val) <= ftol*(fabs(new_obj_val)+fabs(old_obj_val)+EPS));


	//	return new_obj_val;

	//}

	void deep_auto_encoder::encode_hdf_node(H5::Group * group) const
	{
			// Create a fixed-length string
		StrType vls_type(0, 256); // 0 is a dummy argument
		// Open your group
		// Create dataspace for the attribute
		DataSpace att_space(H5S_SCALAR);
		// Create an attribute for the group
		Attribute type_attribute = group->createAttribute("Type",vls_type, att_space);
		// Write data to the attribute
		type_attribute.write(vls_type, "deep_auto_encoder");

		hsize_t structure_dims[1] = { structure.size()};
		DataSpace structure_space( 1, structure_dims );

		DataSet structure_block = group->createDataSet("Structure", PredType::NATIVE_INT,structure_space);

		structure_block.write((void *)&structure[0], PredType::NATIVE_INT, structure_space, structure_space );

		vector<int> neuron_type_int(neuron_types.size());

		for (int i = 0;i< neuron_types.size();i++)
		{
			neuron_type_int[i] = neuron_types[i];
		}

		hsize_t neuron_type_dims[1] = { neuron_types.size()};
		DataSpace neuron_type_space( 1, neuron_type_dims );

		DataSet neuron_type_block = group->createDataSet("NeuronTypes", PredType::NATIVE_INT,neuron_type_space);

		neuron_type_block.write((void *)&neuron_type_int[0], PredType::NATIVE_INT, neuron_type_space, neuron_type_space );

		hsize_t wb_dims[1] = { Wb.size()};
		DataSpace wb_space( 1, wb_dims );

		DataSet wb_block = group->createDataSet("Wb",get_hdf_numeric_datatype<NumericType> (),wb_space);

		EigenVectorType cur_Wb = (EigenVectorType)Wb;

		wb_block.write((void *)cur_Wb.data(), get_hdf_numeric_datatype<NumericType> (), wb_space, wb_space );



	}

	void deep_auto_encoder::decode_hdf_node(const H5::Group * obj)
	{
		Attribute attr = obj->openAttribute("Type");

		string type_str;
		attr.read(attr.getStrType(),type_str);

		if (type_str != "deep_auto_encoder")
			throw runtime_error("Bad HDF Format");

		DataSet structure_block = obj->openDataSet("Structure");

		DataSpace structure_space = structure_block.getSpace();

		int rank = structure_space.getSimpleExtentNdims();

		if (rank != 1)
			throw runtime_error("the rank of the structure space is not equal to 1");

		hsize_t dims_out[1];
		int ndims = structure_space.getSimpleExtentDims( dims_out, NULL);

		int structure_size = dims_out[0];

		structure.resize(structure_size);

		structure_block.read( &structure[0], PredType::NATIVE_INT, structure_space, structure_space );


		DataSet neuron_type_block = obj->openDataSet("NeuronTypes");

		DataSpace neuron_type_space = neuron_type_block.getSpace();

		rank = neuron_type_space.getSimpleExtentNdims();

		if (rank != 1)
			throw runtime_error("the rank of the neuron type is not equal to 1");

		ndims = neuron_type_space.getSimpleExtentDims( dims_out, NULL);

		int neuron_type_num = dims_out[0];

		neuron_types.resize(neuron_type_num);

		vector<int> neuron_type_int(neuron_type_num);

		neuron_type_block.read(& neuron_type_int[0], PredType::NATIVE_INT, neuron_type_space, neuron_type_space );

		for (int i = 0;i< neuron_types.size();i++)
		{
			neuron_types[i] = (neuron_type)neuron_type_int[i];
		}

		num_layers = neuron_types.size();

		if (num_layers != structure.size()-1)
		{
			throw "Bad auto encoder file: dim of structure != dim of neurontypes + 1";
		}

		DataSet wb_block = obj->openDataSet("Wb");

		DataSpace wb_space = wb_block.getSpace();

		rank = wb_space.getSimpleExtentNdims();

		if (rank != 1)
			throw runtime_error("the rank of the wb is not equal to 1");

		ndims = wb_space.getSimpleExtentDims( dims_out, NULL);

		int wb_size = dims_out[0];

		EigenVectorType cur_Wb;

		cur_Wb.resize(wb_size);

		wb_block.read( cur_Wb.data(), get_hdf_numeric_datatype<NumericType> (), wb_space, wb_space );

		Wb = cur_Wb;

		coder_layer_id = (structure.size()-1)/2-1;

		layered_input.resize(num_layers+1);

		init_layered_error.resize(num_layers/2);

		Windex.resize(structure.size()-1);
		bindex.resize(structure.size()-1);

		W.resize(structure.size()-1);
		b.resize(structure.size()-1);

		dW.resize(structure.size()-1);
		db.resize(structure.size()-1);

		int  W_ind = 0;

		for (int level = 0; level < num_layers; level ++)
		{

			Windex[level] = W_ind;
			bindex[level] = W_ind + structure[level] * structure[level + 1];

			W_ind = W_ind + structure[level] * structure[level + 1] + structure[level + 1];
		}

		if (Wb.size() != W_ind)
		{
			throw "Bad auto encoder file: dim of Wb is mistaken";
		}
		dWb.resize(W_ind);

		for (int level = 0; level < num_layers; level ++)
		{
			W[level] = new Map<MatrixType>(Wb.data()+ Windex[level],structure[level + 1],structure[level]);
			b[level] = new Map<VectorType>(Wb.data()+ bindex[level],structure[level + 1]);

			dW[level] = new Map<MatrixType>(dWb.data()+ Windex[level],structure[level + 1],structure[level]);
			db[level] = new Map<VectorType>(dWb.data()+ bindex[level],structure[level + 1]);

		}

		batch_size = 0;
		iter_per_batch = 0;

		best_perf = 0;

		best_machine_file_path = "";

		finetune_iter_num = 0;
	}


//	rapidxml::xml_node<> * deep_auto_encoder::encode_xml_node(rapidxml::xml_document<> & doc) const
//	{
//		using namespace rapidxml;
//
//
//		char * dae_name = doc.allocate_string("deep_auto_encoder");
//		xml_node<> * dae_node = doc.allocate_node(node_element, dae_name);
//
//		char * structure_name = doc.allocate_string("structure");
//		std::ostringstream structure_ss;
////		for_each(structure.begin(),structure.end(),[&structure_ss](int n){structure_ss << n << ' ';});
//		BOOST_FOREACH(int x,structure){structure_ss << x << ' ';}
//		char * structure_value = doc.allocate_string(structure_ss.str().c_str());
//		xml_node<> * structure_node = doc.allocate_node(node_element, structure_name, structure_value);
//
//		char * neurontypes_name = doc.allocate_string("neurontypes");
//		std::ostringstream neurontypes_ss;
////		for_each(neuron_types.begin(),neuron_types.end(),[&neurontypes_ss](neuron_type type){neurontypes_ss << ((type == linear) ? "linear":"logistic") << ' ';});
//		BOOST_FOREACH(neuron_type type,neuron_types){neurontypes_ss << ((type == linear) ? "linear":"logistic") << ' ';}
//		char * neurontypes_value = doc.allocate_string(neurontypes_ss.str().c_str());
//		xml_node<> * neurontypes_node = doc.allocate_node(node_element, neurontypes_name, neurontypes_value);
//
//		std::ostringstream Wb_ss;
//		VectorType curWb = (VectorType)Wb;
//		Wb_ss << curWb;
//		char * Wb_name = doc.allocate_string("Wb");
//		char * Wb_value = doc.allocate_string(Wb_ss.str().c_str());
//		xml_node<> * Wb_node = doc.allocate_node(node_element, Wb_name, Wb_value);
//
//		dae_node->append_node(structure_node);
//		dae_node->append_node(neurontypes_node);
//		dae_node->append_node(Wb_node);
//
//		return dae_node;
//	}
//
//	void deep_auto_encoder::decode_xml_node(rapidxml::xml_node<> & node)
//	{
//
//		using namespace rapidxml;
//
//
//		xml_node<> * structure_node = node.first_node("structure");
//		structure = construct_array<int>(structure_node->value(),boost::is_space() );
//
//		xml_node<> * neurontypes_node = node.first_node("neurontypes");
//		vector<string> neurontypes_str_vec = construct_array<string>(neurontypes_node->value(),boost::is_space() );
//		neuron_types.clear();
//		//for_each(neurontypes_str_vec.begin(),neurontypes_str_vec.end(),
//		//	[this](string type_str)
//		//{
//		//	neuron_type type;
//		//	if (type_str == "linear")
//		//		type = linear;
//		//	else if (type_str == "logistic")
//		//		type = logistic;
//		//	else
//		//		throw "Bad auto encoder file: unrecognized neuron type : " + type_str;
//
//		//	neuron_types.push_back(type);
//		//}
//		//);
//		BOOST_FOREACH(string type_str,neurontypes_str_vec)
//		{
//			neuron_type type;
//			if (type_str == "linear")
//				type = linear;
//			else if (type_str == "logistic")
//				type = logistic;
//			else
//				throw "Bad auto encoder file: unrecognized neuron type : " + type_str;
//
//			neuron_types.push_back(type);
//		}
//		num_layers = neuron_types.size();
//
//		if (num_layers != structure.size()-1)
//		{
//			throw "Bad auto encoder file: dim of structure != dim of neurontypes + 1";
//		}
//
//		xml_node<> * Wb_node = node.first_node("Wb");
//		vector<NumericType> Wb_vec = construct_array<NumericType>(Wb_node->value(),boost::is_space() );
//
//		VectorType cur_Wb;
//		cur_Wb.resize(Wb_vec.size());
//
//		for (int i = 0;i<Wb.size();i++)
//		{
//			cur_Wb(i) = Wb_vec[i];
//		}
//
//		Wb = cur_Wb;
//
//		coder_layer_id = (structure.size()-1)/2-1;
//
//		layered_input.resize(num_layers+1);
//
//		init_layered_error.resize(num_layers/2);
//
//		Windex.resize(structure.size()-1);
//		bindex.resize(structure.size()-1);
//
//		W.resize(structure.size()-1);
//		b.resize(structure.size()-1);
//
//		dW.resize(structure.size()-1);
//		db.resize(structure.size()-1);
//
//		int  W_ind = 0;
//
//		for (int level = 0; level < num_layers; level ++)
//		{
//
//			Windex[level] = W_ind;
//			bindex[level] = W_ind + structure[level] * structure[level + 1];
//
//			W_ind = W_ind + structure[level] * structure[level + 1] + structure[level + 1];
//		}
//
//		if (Wb.size() != W_ind)
//		{
//			throw "Bad auto encoder file: dim of Wb is mistaken";
//		}
//		dWb.resize(W_ind);
//
//		for (int level = 0; level < num_layers; level ++)
//		{
//			W[level] = new Map<MatrixType>(Wb.data()+ Windex[level],structure[level + 1],structure[level]);
//			b[level] = new Map<VectorType>(Wb.data()+ bindex[level],structure[level + 1]);
//
//			dW[level] = new Map<MatrixType>(dWb.data()+ Windex[level],structure[level + 1],structure[level]);
//			db[level] = new Map<VectorType>(dWb.data()+ bindex[level],structure[level + 1]);
//
//		}
//	}
}
