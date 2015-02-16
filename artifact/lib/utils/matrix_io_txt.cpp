#include <iostream>
#include <fstream>
#include <vector>

using namespace std;


#include <boost/lexical_cast.hpp>


#include <artifact/utils/matrix_io_txt.h>
using namespace artifact::utils;

MatrixType artifact::utils::load_matrix_from_txt(const string & file_name)
{

    int cols = 0, rows = 0;
    vector<NumericType> buff;
    buff.reserve(1000000);

    // Read numbers from file into buffer.
    ifstream infile(file_name);
    string line;
    while (getline(infile, line))
    {

        int temp_cols = 0;
        NumericType temp_val = 0.0;
        stringstream stream(line);
        while(stream >> temp_val)
        {
            buff.push_back(temp_val);
            temp_cols ++;
        }

        if (temp_cols == 0)
            continue;
        else if (cols == 0)
            cols = temp_cols;
        else if (temp_cols != cols)
        {
            throw runtime_error("The column number of row " + boost::lexical_cast<string>(rows) + " is not consitent.");
        }

        rows++;
    }

    infile.close();


    // Populate matrix with numbers.
    MatrixType result(rows,cols);
    std::copy(buff.begin(), buff.end(), result.data());

    return result;

}

VectorType artifact::utils::load_vector_from_txt(const string & file_name)
{

    vector<NumericType> buff;
    buff.reserve(1000000);

    // Read numbers from file into buffer.
    ifstream infile(file_name);
    int elem_num = 0;
    NumericType temp_val = 0.0;
    while(infile >> temp_val)
    {
        buff.push_back(temp_val);
        elem_num ++;
    }

    infile.close();


    // Populate matrix with numbers.
    VectorType result(elem_num);
    std::copy(buff.begin(), buff.end(), result.data());

    return result;

}
