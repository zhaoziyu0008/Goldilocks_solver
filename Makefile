CXX = g++
CXXFLAGS = -g -O3 -fopenmp -march=native -std=c++17
LDFLAGS = -g -O3 -fopenmp -pthread

INCLUDES = -I../install/include
LIBDIRS = -L../install/lib
LIBS = -lntl -lgmp -lm

SRCS = poly.cpp fft.cpp 
OBJS = $(SRCS:.cpp=.o) gpu_impl.o
HDRS = resource.hpp manager.hpp log.hpp oracles.hpp poly.h fft.h

CUARCH = -gencode arch=compute_89,code=sm_89
CUSUPP = -Wno-deprecated-gpu-targets

gpu_solver: solver.cpp $(OBJS) 
	nvcc -Xcompiler "$(LDFLAGS)" $^ -lcudart $(CUSUPP) -o $@

%: test_%.cpp $(OBJS) $(HDRS)
	nvcc -Xcompiler "$(LDFLAGS)" $< $(OBJS) $(LIBDIRS) $(LIBS) -lcudart $(CUSUPP) -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $<

gpu_impl.o: gpu_impl.cu gpu_impl.cuh $(HDRS)
	nvcc -O3 -Xcompiler "$(CXXFLAGS)" $(CUARCH) $< -lcudart -lineinfo -c

bench: bench.cu $(HDRS)
	nvcc -O3 -Xcompiler "$(CXXFLAGS)" $(CUARCH) $< -lcudart -lineinfo -o $@

PHONY: clean
clean:
	rm -f $(OBJS) gpu_impl.o ref cm poly gpu_solver
