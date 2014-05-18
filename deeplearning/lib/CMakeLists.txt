
set(srcfiles 
    ./experiment/train_validation_pair.cpp
    ./experiment/experiment_datasets.cpp
    ./experiment/parameter_set.cpp
    ./experiment/data_splitter.cpp
    ./experiment/train_test_pair.cpp
    ./optimization/conjugate_gradient_optimizer.cpp
    ./optimization/optimizer.cpp
    ./optimization/optimize_objective.cpp
    ./optimization/step_update_optimizer.cpp
    ./manifold/isomap.cpp
    ./kernel/linear_kernel.cpp
    ./subspace/LDA.cpp
    ./subspace/svd.cpp
    ./subspace/pca.cpp
    ./subspace/KFDA.cpp
    ./subspace/nmf.cpp
    ./transform/unit_interval_transform.cpp
    ./transform/dataset_transform.cpp
    ./core/serialize.cpp
    ./core/dataset_group.cpp
    ./core/dataset.cpp
    ./core/machine.cpp
    ./core/platform.cpp
    ./core/supervised_dataset.cpp
    ./core/data_splitter.cpp
    ./util/matrix_util.cpp
    ./util/math_util.cpp
    ./classifier/tron.cpp
    ./classifier/linear.cpp
    ./classifier/LeastSquareSVM.cpp
    ./classifier/LibSVMClassifier.cpp
    ./classifier/svm.cpp
    ./classifier/LinearSVMClassifier.cpp
    ./classifier/knn_classifier.cpp
	./classifier/NearestCenterClassifier.cpp
    ./deep/network_objective.cpp
    ./deep/restricted_boltzmann_machine.cpp
    ./deep/deep_auto_encoder.cpp
    ./deep/initializer/random_layerwise_initializer.cpp
    ./deep/initializer/rbm_layerwise_initializer.cpp
    ./deep/initializer/ae_layerwise_initializer.cpp
    ./deep/network_optimobj_adapter.cpp
    ./deep/objective/combined_objective.cpp
    ./deep/objective/fisher_objective.cpp
	./deep/objective/lssvm_objective.cpp
    ./deep/objective/mse_objective.cpp
    ./deep/objective/MaximumMarginObjective.cpp
    ./deep/objective/ridge_regression_regularizor.cpp
    ./deep/objective/data_related_network_objective.cpp
    ./deep/objective/cross_entropy_objective.cpp
    ./deep/objective/self_related_network_objective.cpp
    ./classifier/blas/dnrm2.c
    ./classifier/blas/daxpy.c
    ./classifier/blas/ddot.c
    ./classifier/blas/dscal.c
)








IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
			SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_FULLY_DYNAMIC_STRING -std=c++0x -O2")
			SET(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)
			include_directories(${CMAKE_SOURCE_DIR}/include /opt/local/include /opt/local/include/eigen3)
			link_directories(/opt/local/lib)
			add_library(learning SHARED ${srcfiles}) 
			target_link_libraries(learning hdf5 hdf5_cpp camp log4cpp boost_filesystem boost_system)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
			include_directories(${CMAKE_SOURCE_DIR}/include D:/Lib/include D:/Lib/include/hdf5 D:/Lib/include/camp D:/Lib/include/log4cpp)
			link_directories(D:/Lib/lib/boost D:/Lib/lib/camp D:/Lib/lib/hdf5 D:/Lib/lib/log4cpp)
			add_library(learning SHARED ${srcfiles}) 
			target_link_libraries(learning hdf5 hdf5_cpp camp log4cpp)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")



