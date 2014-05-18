#ifndef NEURON_LAYER_H
#define NEURON_LAYER_H



namespace deeplearning
{

	class neuron_layer
	{
	public:
		virtual MatrixType output(const MatrixType & input) = 0;
		virutal VectorType output(const VectorType & input) = 0;
	};

}



#endif
