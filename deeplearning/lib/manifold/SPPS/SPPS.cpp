// SPPS.cpp : 定义控制台应用程序的入口点。
//


#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <ctime>
#include <cstdlib>
using namespace std;

#include <blitz/array.h>
using namespace blitz;

#include "blitzutil.h"
#include "spps.h"


int main(int argc, char *argv[])
{

	ifstream param("Param.ini");

	string datafilename, initSfilename, initTfilename, initSigmafilename, neighborfilename;

	param >> datafilename>>  initSfilename>> initTfilename >> initSigmafilename >> neighborfilename;


	Array< psFloat, 2 > data =  read_column_major_matrix<psFloat>(datafilename);
	Array<psFloat,2> initS =  read_column_major_matrix<psFloat>(initSfilename);
	Array< psFloat, 2 > initT =  read_column_major_matrix<psFloat>(initTfilename);
	Array<psFloat,1> initSigma =  read_matrix<psFloat,1>(initSigmafilename);
	Array< int, 2 > neighbor =  read_column_major_matrix<int >(neighborfilename);

	Array<psFloat,2> projY  = make_column_major_matrix<psFloat>(initS.rows(), data.columns());

	Array<psFloat,2> mX = make_column_major_matrix<psFloat>(data.rows(), data.columns());

	SPPS_init(data.data(), data.rows(), data.columns(), initS.data(), initT.data(), initS.rows(), initT.columns(), Gaussian, initSigma.data(), neighbor.data(), neighbor.columns());

	slps_map_train(mX.data()); 

	slps_project_train(projY.data());

	SPPS_final();

	ofstream mapFile("MapFile.txt");

	mapFile << mX;

	ofstream projFile("ProjFile.txt");

	projFile << projY;


	return EXIT_SUCCESS;
}
