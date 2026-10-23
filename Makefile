all:
	g++ -o nnfc main.cpp fc_layer.cpp -Wall -pedantic -O2 -lpthread -mavx -mavx2 -mfma

