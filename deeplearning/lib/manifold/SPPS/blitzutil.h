#ifndef BLITZUTIL_H_
#define BLITZUTIL_H_

#include <fstream>
using namespace std;

#include <blitz/array.h>
using namespace blitz;



template<typename T_numtype>
Array<T_numtype, 2 > transpose_storage_order(const Array<T_numtype, 2 > & src)
{
	Array<T_numtype, 2 > dest(src.columns(),src.rows());

	for (int i = 0; i < src.columns();i++)
	{
		for (int j = 0; j < src.rows();j++)
		{
			dest(i,j) = src(j,i);
		}

	}

	return dest.transpose(1,0);
}

template<typename T_numtype, int N>
Array< T_numtype, N > read_matrix(const string & filename)
{
	Array< T_numtype,N> init;
	ifstream initfile(filename.c_str());
	initfile >> init;

	return init;


}



template<typename T_numtype>
Array< T_numtype, 2 > read_column_major_matrix(const string & filename)
{
	Array< T_numtype,2> init;
	ifstream initfile(filename.c_str());
	initfile >> init;

	return transpose_storage_order(init);


}

template<typename T_numtype>
Array< T_numtype, 2 > make_column_major_matrix(int row, int column)
{
	Array<T_numtype,2> init(column,row);

	return init.transpose(1,0);
}


#endif