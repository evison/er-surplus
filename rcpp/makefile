# consumer surplus maximizer
CSMax: CSMaxSA.o
	g++  -std=gnu++11 -g -O0  -larmadillo   -o CSMaxSA CSMaxSA.o

CSMaxSA.o: CSMaxSA.cpp 
	g++  -std=gnu++11 -g -O0 -I.  -fpic  -march=x86-64 -mtune=generic -pipe -fstack-protector-strong --param=ssp-buffer-size=4  -c CSMaxSA.cpp


### social surplus maximizer
SSMax: SSMax.o
	g++ -std=gnu++11 -g -O0  -larmadillo   -o SSMax SSMax.o

SSMax.o: SSMax.cpp 
	g++ -std=gnu++11 -g -O0 -I.  -fpic  -march=x86-64 -mtune=generic -pipe -fstack-protector-strong --param=ssp-buffer-size=4  -c SSMax.cpp


# CF
CFSGD: CFSGD.o
	g++ -std=gnu++11  -g -O0  -larmadillo   -o CFSGD CFSGD.o
CFSGD.o: CFSGD.cpp 
	g++ -std=gnu++11 -g -O0 -I.  -fpic  -march=x86-64 -mtune=generic -pipe -fstack-protector-strong --param=ssp-buffer-size=4  -c CFSGD.cpp


# model purchase probability (0/1 case)
ZeroOneMax: ZeroOneMax.o
	g++ -std=gnu++11 -g -O0  -larmadillo   -o ZeroOneMax ZeroOneMax.o
ZeroOneMax.o: ZeroOneMax.cpp 
	g++ -std=gnu++11 -g -O0 -I.  -fpic  -march=x86-64 -mtune=generic -pipe -fstack-protector-strong --param=ssp-buffer-size=4  -c ZeroOneMax.cpp


## social surplus evaluation

# CF
SSEval: SSEval.o
	g++ -std=gnu++11  -g -O0  -larmadillo   -o SSEval SSEval.o
SSEval.o: SSEval.cpp 
	g++ -std=gnu++11 -g -O0 -I.  -fpic  -march=x86-64 -mtune=generic -pipe -fstack-protector-strong --param=ssp-buffer-size=4  -c SSEval.cpp
