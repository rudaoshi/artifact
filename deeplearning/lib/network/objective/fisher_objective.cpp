
#include <liblearning/deep/objective/fisher_objective.h>

#include <liblearning/core/supervised_dataset.h>
#include <liblearning/deep/deep_auto_encoder.h>


using namespace deep;
using namespace deep::objective;
fisher_objective::fisher_objective()
{
	type = encoder_related;

}

fisher_objective::~fisher_objective()
{
}



void fisher_objective::set_dataset(const shared_ptr<dataset> & data_set_)
{
//	std::cout << "Setting New Data Set For Fisher Objective" <<std::endl;
	if (data_set  && *data_set == *data_set_)
	{
		//std::cout << "Identical with Previous Data Set, Need not Do Anything" <<std::endl;
		//std::cout << "Previous Data Set Size = " << (data_set == 0 ? 0 : data_set->get_sample_num()) <<std::endl;
		//std::cout << "New Data Set Size = " << data_set_->get_sample_num() <<std::endl;
		return ;
	}

//	std::cout << "Setting New Data Set With Sample Size" << data_set_->get_sample_num() <<std::endl;
	data_set = data_set_;

	const supervised_dataset & s_data_set = dynamic_cast<const supervised_dataset &>(*data_set);

	int sample_num = s_data_set.get_sample_num();
	int class_num = s_data_set.get_class_num();

	const vector<int> & label = s_data_set.get_label();
	const vector<int> & class_id = s_data_set.get_class_id();
	const vector<int> & class_elem_num = s_data_set.get_class_elem_num();

	Aw.resize(sample_num,sample_num);
	Ab.resize(sample_num,sample_num);

	EigenMatrixType Aw_temp(sample_num,sample_num);
	EigenMatrixType Ab_temp(sample_num,sample_num);

    int i, j,k;

    for (i = 0;i<sample_num;i++)
    {
        for (j = 0;j<sample_num;j++)
        {
            if (label[i] == label[j])
            {
                for (k = 0;k<class_num;k++)
                {
                    if (class_id[k] == label[i])
                        break;
                }
                NumericType nc = class_elem_num[k];
                Aw_temp(i,j) = 1.0/nc;
                Ab_temp(i,j) = 1.0/sample_num - 1/nc;
            }
            else
            {
                Aw_temp(i,j) = 0;
                Ab_temp(i,j) = 1.0/sample_num;
            }
        }
    }

	EigenMatrixType Aw_diff_helper_temp;
	Aw_diff_helper_temp = Aw_temp + Aw_temp.transpose();
	EigenVectorType Aw_diag_temp = Aw_diff_helper_temp.colwise().sum();
	Aw_diag_temp.asDiagonal().subTo(Aw_diff_helper_temp);

	EigenMatrixType Ab_diff_helper_temp;
	Ab_diff_helper_temp = Ab_temp + Ab_temp.transpose();
	EigenVectorType Ab_diag_temp = Ab_diff_helper_temp.colwise().sum();
	Ab_diag_temp.asDiagonal().subTo(Ab_diff_helper_temp);

	Aw = Aw_temp;
	Ab = Ab_temp;

	Aw_diff_helper = Aw_diff_helper_temp;
	Ab_diff_helper = Ab_diff_helper_temp;



}

#include <liblearning/util/matrix_util.h>

#include <limits>


#include <boost/math/special_functions/fpclassify.hpp>




NumericType fisher_objective::prepared_value(deep_auto_encoder & net) 
{
//	std::cout << "Fisher Objective: Get Current Feature " <<std::endl;
	const MatrixType & feature = net.get_layered_output(net.get_coder_layer_id());
//	std::cout << "Fisher Objective Got " <<std::endl;

//#if defined(USE_PARTIAL_GPU)
//
//	
//	GPUMatrixType gM; 
//	{
//		GPUMatrixType gFeature = feature;
//		std::cout << "Computing Squared Dist between features with size [" << feature.rows() << "X" << feature.cols() << "]" <<std::endl;
//
//		gM = sqdist(gFeature,gFeature);
//		std::cout << "Squared Dist between features Computed " <<std::endl;
//	}
//
//	{
//		GPUMatrixType gAw = Aw;
//		std::cout << "Computing trSw with Aw size  [" << Aw.rows() << "X" << Aw.cols() << "]" <<std::endl;
//		trSw = 0.5*(gAw.array()*gM.array()).sum();
//		std::cout << "trSw Computed =" << trSw <<std::endl;
//	}
//
//	{
//		GPUMatrixType gAb = Ab;
//
//		std::cout << "Computing trSb with Ab size  [" << Ab.rows() << "X" << Ab.cols() << "]" <<std::endl;
//		trSb = 0.5*(gAb.array()*gM.array()).sum();// + std::numeric_limits<NumericType>::epsilon();
//		std::cout << "trSb Computed =" << trSb <<std::endl;
//	}
//
//#else
//	std::cout << "Computing Squared Dist between features with size [" << feature.rows() << "X" << feature.cols() << "]" <<std::endl;
	MatrixType M = sqdist(feature,feature);
//	std::cout << "Squared Dist between features Computed " <<std::endl;
    
//	std::cout << "Computing trSw with Aw size  [" << Aw.rows() << "X" << Aw.cols() << "]" <<std::endl;
	trSw = 0.5*(Aw.array()*M.array()).sum();// + std::numeric_limits<NumericType>::epsilon();
//	std::cout << "trSw Computed =" << trSw <<std::endl;


//	std::cout << "Computing trSb with Ab size  [" << Ab.rows() << "X" << Ab.cols() << "]" <<std::endl;
    trSb = 0.5*(Ab.array()*M.array()).sum();// + std::numeric_limits<NumericType>::epsilon();
//	std::cout << "trSb Computed =" << trSb <<std::endl;
//#endif

	if (trSw <= 0.0)
	{
		trSw = 0.0;
		return 0.0;
	}



	if (trSb <= 0)
	{
		trSw = trSb = 0.0;
		return 0.0;
	}
    NumericType value = trSw / trSb;

	if (value < 0)
	{
		std::cout << "Error Occured!" << std::endl;
		net.save_hdf("bad_machine2.hdf");

		trSw = trSb = 0.0;
		return 0.0;
	}
	else if (boost::math::isnan(value))
	{
		std::cout << "Nan Error Occured!" << std::endl;
		net.save_hdf("bad_machine_nan.hdf");

		trSw = trSb = 0.0;
		return 0.0;
	}
	else if(boost::math::isinf(value))
	{
		std::cout << "Inf Error Occured!" << std::endl;
		net.save_hdf("bad_machine_inf.hdf");

		trSw = trSb = 0.0;
		return 0.0;
	}
	
	current_obj_val = value;
	return value;

}


vector<shared_ptr<MatrixType>> fisher_objective::prepared_value_delta(deep_auto_encoder & net) 
{
//	std::cout << "Fisher Objective: Get Current Feature " <<std::endl;
	const MatrixType & feature = net.get_layered_output(net.get_coder_layer_id());

//	std::cout << "Fisher Objective Got " <<std::endl;

	shared_ptr<MatrixType> JF;
	if (trSw == 0.0)
	{
//		std::cout << "Zero SW OCCURED, return Zero Diff ! " <<std::endl;
		JF.reset(new MatrixType(MatrixType::Zero(feature.rows(),feature.cols())));
//		std::cout << "Zero SW OCCURED, Zero Diff returned " <<std::endl;
		
	}
	else
	{
//		std::cout << "Non-Zero SW , Computing JSw " <<std::endl;
		//MatrixType Aw_AwT = Aw + Aw.transpose();
		//VectorType Aw_diag = Aw_AwT.colwise().sum();
		//Aw_diag.asDiagonal().subTo(Aw_AwT);
		MatrixType JSw = -feature*Aw_diff_helper;  
//		std::cout << "JSw Computed" <<std::endl;

//		std::cout << "Computing JSb " <<std::endl;
		//MatrixType Ab_AbT = Ab + Ab.transpose();
		//VectorType Ab_diag = Ab_AbT.colwise().sum();
		//Aw_diag.asDiagonal().subTo(Aw_AwT);
		MatrixType JSb = -feature*Ab_diff_helper;  

//		std::cout << "JSb Computed" <<std::endl;

//		std::cout << "trSW = " <<trSw <<" trSb = " <<trSb<<std::endl;

		JF.reset(new MatrixType((JSw-JSb*(trSw/trSb))/trSb)); 

//		std::cout << "JF Computed" <<trSb<<std::endl;
	}


	if (net.get_neuron_type_of_layer(net.get_coder_layer_id()) == logistic)
	{
		std::cout << "this should not happen" << std::endl;
		*JF = net.error_diff_to_delta(*JF, net.get_coder_layer_id());
	}

	vector<shared_ptr<MatrixType>> result(2);

	result[0] = JF ;
	result[1] = shared_ptr<MatrixType>();
	return result;
}


fisher_objective * fisher_objective::clone()
{
	return new fisher_objective(*this);
}

string fisher_objective::get_info()
{
	string info = "Fisher Objective Value = " + boost::lexical_cast<string>(current_obj_val);

	info += " where trSw = " + boost::lexical_cast<string>(trSw) ;

	info += " and trSb = " + boost::lexical_cast<string>(trSb) ;

	return info;

}
