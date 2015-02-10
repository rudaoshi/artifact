#include <liblearning/subspace/KFDA.h>


using namespace subspace;

#if defined USE_MATLAB
#include <engine.h>
#endif

#include <liblearning/core/platform.h>


KFDA::KFDA()
{
}


void KFDA::train(const shared_ptr<dataset>  & train_)
{
	train_set = dynamic_pointer_cast<supervised_dataset>( train_);

#if defined USE_MATLAB
	Engine *ep = platform::Instance().get_matlab_eigen();

	if (ep == 0)
		throw runtime_error("Matlab Engine not Prepared!");
		
	mxArray *TrainKernel = NULL, *TrainLabel = NULL;
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

	MatrixType train_data = train_set->get_data();

	EigenMatrixType train_kernel_matrix = param->kernelfunc->eval(train_data,train_data);

	/* 
	 * Create a variable for our data
	 */
	TrainKernel = mxCreateDoubleMatrix(train_kernel_matrix.rows(),train_data.cols(), mxREAL);
	memcpy((void *)mxGetPr(TrainKernel), (void *)train_kernel_matrix.data(), train_kernel_matrix.size()*sizeof(double));

	TrainLabel = mxCreateDoubleMatrix(train_set->get_sample_num(),1, mxREAL);
	memcpy((void *)mxGetPr(TrainLabel), (void *)&train_set->get_label()[0], train_set->get_sample_num()*sizeof(double));

	/*
	 * Place the variable TrainData into the MATLAB workspace
	 */
	engPutVariable(ep, "TrainKernel", TrainKernel);

	engPutVariable(ep, "TrainLabel", TrainLabel);


	/*
	 * Evaluate a function of time, distance = (1/2)g.*t.^2
	 * (g is the acceleration due to gravity)
	 */


	engEvalString(ep, "cd('D:\\Sun\\WorkSpace\\liblearning.mat\\kernel');");
	engEvalString(ep, "option.Kernel = 1; eigvector = KDA(option,TrainLabel,TrainKernel);");

	if ((EigenVector = engGetVariable(ep,"eigvector")) == NULL)
	      throw std::runtime_error("Matlab KDA routine runs failed!");

	EigenMatrixType eigenvector_temp(mxGetM(EigenVector),mxGetN(EigenVector));
	memcpy((void *)eigenvector_temp.data(), (void *)mxGetPr(EigenVector),eigenvector_temp.size()*sizeof(double));
	
	eigenvector = eigenvector_temp;


	mxDestroyArray(TrainKernel);
	mxDestroyArray(TrainLabel);
	mxDestroyArray(EigenVector);
	
	//engClose(ep);
#endif
}

shared_ptr<dataset> KFDA::extract_feature(const shared_ptr<dataset>  & data)
{
	MatrixType kernel_matrix = param.kernelfunc->eval(data->get_data(),train_set->get_data());

	MatrixType feature = (kernel_matrix*eigenvector).transpose();

	return data->clone_update_data(feature);
}
