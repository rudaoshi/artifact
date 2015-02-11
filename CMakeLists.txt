
set(srcfiles 
    ./artifact/lib/experiment/train_validation_pair.cpp
    ./artifact/lib/experiment/experiment_datasets.cpp
    ./artifact/lib/experiment/parameter_set.cpp
    ./artifact/lib/experiment/data_splitter.cpp
    ./artifact/lib/experiment/train_test_pair.cpp
    ./artifact/lib/optimization/conjugate_gradient_optimizer.cpp
    ./artifact/lib/optimization/optimizer.cpp
    ./artifact/lib/optimization/optimize_objective.cpp
    ./artifact/lib/optimization/step_update_optimizer.cpp
    ./artifact/lib/manifold/isomap.cpp
    ./artifact/lib/kernel/linear_kernel.cpp
    ./artifact/lib/subspace/LDA.cpp
    ./artifact/lib/subspace/svd.cpp
    ./artifact/lib/subspace/pca.cpp
    ./artifact/lib/subspace/KFDA.cpp
    ./artifact/lib/subspace/nmf.cpp
    ./artifact/lib/transform/unit_interval_transform.cpp
    ./artifact/lib/transform/dataset_transform.cpp
    ./artifact/lib/core/serialize.cpp
    ./artifact/lib/core/dataset_group.cpp
    ./artifact/lib/core/dataset.cpp
    ./artifact/lib/core/machine.cpp
    ./artifact/lib/core/platform.cpp
    ./artifact/lib/core/supervised_dataset.cpp
    ./artifact/lib/core/data_splitter.cpp
    ./artifact/lib/util/matrix_util.cpp
    ./artifact/lib/util/math_util.cpp
    ./artifact/lib/classifier/tron.cpp
    ./artifact/lib/classifier/linear.cpp
    ./artifact/lib/classifier/LeastSquareSVM.cpp
    ./artifact/lib/classifier/LibSVMClassifier.cpp
    ./artifact/lib/classifier/svm.cpp
    ./artifact/lib/classifier/LinearSVMClassifier.cpp
    ./artifact/lib/classifier/knn_classifier.cpp
	./artifact/lib/classifier/NearestCenterClassifier.cpp
	artifact/lib/network/__depredicated/linear_layer.cpp
	artifact/lib/network/__depredicated/logistic_layer.cpp
	./artifact/lib/network/layer/mlp_layer.cpp
    artifact/lib/network/__depredicated/network_objective.cpp
    ./artifact/lib/network/restricted_boltzmann_machine.cpp
    artifact/lib/network/__depredicated/deep_auto_encoder.cpp
    ./artifact/lib/network/deep_network.cpp
    ./artifact/lib/network/initializer/random_layerwise_initializer.cpp
    ./artifact/lib/network/initializer/rbm_layerwise_initializer.cpp
    ./artifact/lib/network/initializer/ae_layerwise_initializer.cpp
    artifact/lib/network/__depredicated/network_optimobj_adapter.cpp
    ./artifact/lib/network/objective/combined_objective.cpp
    ./artifact/lib/network/objective/fisher_objective.cpp
	./artifact/lib/network/objective/lssvm_objective.cpp
    ./artifact/lib/network/objective/mse_objective.cpp
    ./artifact/lib/network/objective/MaximumMarginObjective.cpp
    ./artifact/lib/network/objective/ridge_regression_regularizor.cpp
    ./artifact/lib/network/objective/data_related_network_objective.cpp
    ./artifact/lib/network/objective/cross_entropy_objective.cpp
    ./artifact/lib/network/objective/self_related_network_objective.cpp
    ./artifact/lib/classifier/blas/dnrm2.c
    ./artifact/lib/classifier/blas/daxpy.c
    ./artifact/lib/classifier/blas/ddot.c
    ./artifact/lib/classifier/blas/dscal.c
)








IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
			SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_FULLY_DYNAMIC_STRING -std=c++0x -O2")
			SET(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)
			include_directories(${CMAKE_SOURCE_DIR}/artifact/include /usr/local/include /usr/local/include/eigen3)
			link_directories(/usr/local/lib)
			add_library(learning SHARED ${srcfiles}) 
			target_link_libraries(learning hdf5 hdf5_cpp camp log4cpp boost_filesystem boost_system)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
			include_directories(${CMAKE_SOURCE_DIR}/include D:/Lib/include D:/Lib/include/hdf5 D:/Lib/include/camp D:/Lib/include/log4cpp)
			link_directories(D:/Lib/lib/boost D:/Lib/lib/camp D:/Lib/lib/hdf5 D:/Lib/lib/log4cpp)
			add_library(learning SHARED ${srcfiles}) 
			target_link_libraries(learning hdf5 hdf5_cpp camp log4cpp)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")



