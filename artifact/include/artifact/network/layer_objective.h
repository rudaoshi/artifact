#ifndef LAYER_OBJECTIVE_H
#define LAYER_OBJECTIVE_H


namespace core
{
  class layer_objective
  {
  public:

      virtual double value(const neuron_layer & layer);

      virtual VectorType diff(const neuron_layer & layer);
  };

}


#endif
