default:
	nvcc Sgemm_v4.cu -o gemm.out -arch=compute_70 -code=compute_70 --ptxas-options=-v

run:
	./gemm.out