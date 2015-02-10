#include <liblearning/subspace/LDA.h>


using namespace subspace;
#if defined USE_MATLLAB
#include <engine.h>
#endif
#include <liblearning/core/platform.h>


LDA::LDA()
{
}


void LDA::train(const shared_ptr<dataset>  & train_)
{
	train_set = dynamic_pointer_cast<supervised_dataset>( train_);

#if defined USE_MATLAB
	Engine *ep = platform::Instance().get_matlab_eigen();

	if (ep == 0)
		throw runtime_error("Matlab Engine not Prepared!");
		
	mxArray *TrainData = NULL, *TrainLabel = NULL;
	mxArray *EigenVector = 0;


	//if (!(ep = engOpen("\0"))) {
	//	throw std::runtime_error("Can't start MATLAB engine\n");
	//}

	/*
	 * PART I
	 *
	 * For the first half of this demonstration, we will send data
	 * to MATLAB, analyze the data, and plot the result.
	 */

	EigenMatrixType train_data = train_set->get_data();

	/* 
	 * Create a variable for our data
	 */
	TrainData = mxCreateDoubleMatrix(train_data.rows(),train_data.cols(), mxREAL);
	memcpy((void *)mxGetPr(TrainData), (void *)train_data.data(), train_data.size()*sizeof(double));

	TrainLabel = mxCreateDoubleMatrix(train_set->get_sample_num(),1, mxREAL);
	memcpy((void *)mxGetPr(TrainLabel), (void *)&train_set->get_label()[0], train_set->get_sample_num()*sizeof(double));

	/*
	 * Place the variable TrainData into the MATLAB workspace
	 */
	engPutVariable(ep, "TrainData", TrainData);

	engPutVariable(ep, "TrainLabel", TrainLabel);


	/*
	 * Evaluate a function of time, distance = (1/2)g.*t.^2
	 * (g is the acceleration due to gravity)
	 */

	engEvalString(ep, "cd('D:\\Sun\\WorkSpace\\liblearning.mat\\linear');");

	string evalstr;
	if (param->reg == 0)
	{
		evalstr = "option = []; eigvector = LDA(TrainLabel,option,TrainData');";
	}
	else
	{
		evalstr =  "option.Regu = 1; option.ReguAlpha=" +  boost::lexical_cast<string>(param->reg) + "; eigvector = LDA(TrainLabel,option,TrainData');";
	}
//	engEvalString(ep, "option.Regu = 1; eigvector = LDA(TrainLabel,option,TrainData');");
	engEvalString(ep, evalstr.c_str());

	if ((EigenVector = engGetVariable(ep,"eigvector")) == NULL)
	      throw std::runtime_error("Matlab KDA routine runs failed!");

	EigenMatrixType eigenvector_temp(mxGetM(EigenVector),mxGetN(EigenVector));
	memcpy((void *)eigenvector_temp.data(), (void *)mxGetPr(EigenVector),eigenvector_temp.size()*sizeof(double));
	
	eigenvector = eigenvector_temp;


	mxDestroyArray(TrainData);
	mxDestroyArray(TrainLabel);
	mxDestroyArray(EigenVector);

#endif
	//engClose(ep);

}

shared_ptr<dataset> LDA::extract_feature(const shared_ptr<dataset>  & data)
{
	
	MatrixType feature = (data->get_data().transpose()*eigenvector).transpose();

	return data->clone_update_data(feature);
}
