#ifndef PCA_H

#define PCA_H


#include <liblearning/core/config.h>


namespace subspace
{
	class pca
	{
		VectorType mean;
		MatrixType P;
		VectorType eigenval;

	protected:

		void pca_auto(const MatrixType & centered_train_);

		void pca_std(const MatrixType & centered_train_);

		void pca_trans(const MatrixType & centered_train_);

	public:

		pca (const MatrixType & train_);

		MatrixType apply(const MatrixType & test, int k);
	};
}

#endif