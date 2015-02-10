#ifndef UTIL_H_
#define UTIL_H_

void column_wise_add(psFloat* B, psFloat alpha, psFloat* A, int r, int c, psFloat beta, psFloat* x);

void column_wise_sum(psFloat* n, psFloat* A, int r, int c);

void column_wise_norm2(psFloat* n, psFloat* A, int r, int c);

void column_wise_distance_mv(psFloat* n, psFloat* A, int r, int c,psFloat* x);

void column_wise_squared_distance_mm(psFloat* n, psFloat* A, psFloat* B, int r, int c);

void column_wise_scal(psFloat* B,psFloat alpha, psFloat* A, int r, int c, psFloat* x);

void column_wise_normal(psFloat* B,psFloat* A, int r, int c, psFloat* x);

void column_wise_dot(psFloat* n, psFloat* A, psFloat * B, int r, int c);

void column_wise_squared_norm2(psFloat* n, psFloat* A, int r, int c);

void element_wise_scal(psFloat* C,psFloat alpha, psFloat* A, psFloat * B, int r, int c);

void fill(psFloat* a, unsigned int N, unsigned int inc, psFloat val);

void display(char * msg, psFloat * dm, int r , int c);

/**
* �������ref������queryÿһ��֮���ƽ������
*/
void pairwise_squared_distance(psFloat* ref, int wR, int pR, psFloat* query, int wQ, int pQ, int dim,  psFloat* RQ);

/**
*  ������A����index��ָ�����п�����X��
*/
void copy_indexed_columns(psFloat * X, int * index, int nX, psFloat * A, int r, int c);

#endif