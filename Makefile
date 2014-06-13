CXX=g++

ASMSRCS := gemm_float.asm
ASMOBJS := $(subst .asm,.do,$(ASMSRCS))

CPPSRCS := $(shell find . -name '*.cpp' -not -name '._*')
CPPOBJS := $(subst .cpp,.o,$(CPPSRCS))


CPPFLAGS=-Ofast -DTARGET_PI -march=armv6 \
-mfloat-abi=hard \
-ftree-vectorize \
-funroll-all-loops \
-mfpu=vfp \

%.cdat: %.asm helpers.asm
	m4 $< | qpu-asm -o $(basename $@).cdat -c g_$(basename $@)Code

%.do: %.cdat
	$(CXX) $(CPPFLAGS) -x c -c $< -o $(basename $@).do	

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -fPIC -c $< -o $(basename $@).o

gemm: $(CPPOBJS) $(ASMOBJS)
	g++ -g -O3 -o gemm $(CPPOBJS) $(ASMOBJS) -lblas
