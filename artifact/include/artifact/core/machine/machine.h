#ifndef MACHINE_H_
#define MACHINE_H_

#include <deeplearning/core/data/data_set.h>

namespace deeplearning
{
	/* machine_maker class.
	 *
	 * machine_maker builds machines from input data. All training algorithms
	 * in this library are subclass of machine_maker
	 */
	template<typename Machine>
	class machine_maker
	{
	public:

		virtual Machine train(
			const typename sample_set_type<Machine::sample_type>::type & train_data
			) = 0;

	};

	template<typename SampleType, typename OutputType>
	class machine
	{

	public:

		typedef SampleType sample_type;
		typedef OutputType output_type;

	public:

		virtual OutputType predict(const SampleType & testdata) = 0;
		virtual typename sample_set_type<OutputType>::type
					predict(const typename sample_set_type<SampleType>::type & test_set) = 0;

	};


	template<typename ParameterType>
	class parameterized
	{


	public:

		virtual const ParameterType  & get_parameter() = 0;

		virtual void set_parameter(const ParameterType& parameter_) = 0
	};






}
#endif
