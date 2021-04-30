CC:=mpic++

#TFPREFIX:=/home/limingfan/tf_cpp_15/tf_root
#INC_DIR:=  -I$(TFPREFIX)/include 
#LINK_DIR:= -L$(TFPREFIX)/lib -L/home/limingfan/local/scalapack/lib -L/home/limingfan/intel/mkl/lib/intel64 #-L/home/limingfan/local/lapack

LINK_LIB:=-ltensorflow_cc -ltensorflow_framework -lscalapack -lmkl_rt
CFLAGS:=-std=c++11 -g -Wall -fPIC -fopenmp

train: train.o scalapack.o
	$(CC) $^ -o $@  $(LINK_LIB)
train.o: train-mkl.cc
	$(CC) -c -o $@ $< $(CFLAGS)
scalapack.o: scalapack.cc
	$(CC) -c -o $@ $< $(CFLAGS)
all: train
clean:
	rm train *.o
