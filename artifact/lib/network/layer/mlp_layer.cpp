#include <algorithm>

#include <artifact/network/layer/activator.h>
#include <artifact/network/layer/mlp_layer.h>


using namespace artifact::network;


int mlp_layer::get_input_dim() const
{
	return input_dim;
}


int mlp_layer::get_output_dim() const
{
	return output_dim;
}


mlp_layer::mlp_layer(int input_dim_, int output_dim_, shared_ptr<activator>  active_func_)
    : input_dim(input_dim_), output_dim(output_dim_), active_func(active_func_)
{

}



MatrixType mlp_layer::predict(const MatrixType & X)
{

    MatrixType act_val = W*X;

    act_val.colwise() += b;
    return active_func->activate(act_val);


}

VectorType mlp_layer::predict(const VectorType & x)
{

    VectorType act_val = W*x;

    act_val.colwise() += b;
    return active_func->activate(act_val);

}


pair<MatrixType, MatrixType> mlp_layer::predict_with_activator(const MatrixType & X)
{

    MatrixType act_val = W*X;

    act_val.colwise() += b;
    return make_pair(act_val, active_func->activate(act_val));


}


MatrixType mlp_layer::compute_delta(
        const MatrixType & activator,
        const MatrixType & output,
        const VectorType & y)
{
    //output_delta = 2/N*((Reconstruction - X).*Reconstruction.*(1-Reconstruction));

    if (not loss_func)
    {
        throw runtime_error("The layer is not assigned with an objective.");
    }
    return active_func->gradient(activator) * loss_func->gradient(output, y);
}

MatrixType mlp_layer::backprop_delta(const MatrixType & delta,
        const MatrixType & activator)
{

    MatrixType new_delta =  (W.transpose()*delta);
    new_delta.array() *= active_func->gradient(activator).array();

    return new_delta;

}

bool mlp_layer::is_loss_contributor() const
{
    return bool(loss_func);
}

pair<MatrixType, VectorType> mlp_layer::compute_param_gradient(const MatrixType & input, const MatrixType & delta)
{
    //dWb_mat = delta*[self.layered_output{2*num_maps}', ones(N,1)];

#if defined USE_PARTIAL_GPU
	GPUMatrixType gDelta = delta, gInput = input, gDiffw;
	GPUVectorType gDiffb;

	gDiffw = gDelta*gInput.transpose();
	gDiffb = gDelta.rowwise().sum();

	MatrixType diff_W = (MatrixType) gDiffw;
	VectorType diff_b = (VectorType) gDiffb;

#else
    MatrixType diff_W = delta*input.transpose();
    VectorType diff_b = delta.rowwise().sum();
#endif

    return make_pair(diff_W, diff_b);

}


void mlp_layer::set_loss(const shared_ptr<loss_function> & loss_func_ )
{
    loss_func = loss_func_;
}
