CC:=gcc
NVCC:=nvcc

program: matrixNorm.c matrixNorm.cu

	 $(CC) matrixNorm.c -lm -o norm.out

	 $(NVCC) matrixNorm.cu -lm --expt-relaxed-constexpr -o gpunorm.out

clean:
	rm -rf *.out