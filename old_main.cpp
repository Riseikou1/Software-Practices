#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <ctime>
#include <chrono>

extern void fc_layer(size_t data_cnt, size_t input_dim, size_t output_dim, float* matrix, float* bias, float* input, float* output, int threads);

int 
main(int argc, char** argv) {
	size_t model_dim_in = 4096;
	size_t model_dim_out = 4096;
	size_t maximum_inputs = 4096;
	int threads = 1;
	if ( argc < 3 ) {
		fprintf(stderr, "error: input output filename arguments not provided\n" );
		exit(10);
	}
	char* fname_input = argv[1];
	char* fname_output = argv[2];

	if ( argc >= 4 ) {
		int threadcnt = atoi(argv[3]);
		if ( threadcnt > 1 && threadcnt < 13 ) threads = threadcnt;
	}
	printf( "Requesting %d threads\n", threads );


	float* model_matrix = (float*)aligned_alloc(64,sizeof(float)*model_dim_in*model_dim_out);
	float* model_bias = (float*)aligned_alloc(64,sizeof(float)*model_dim_out);
	FILE* f_model_matrix = fopen("vgg19.w24.matrix.bin", "rb");
	FILE* f_model_bias = fopen("vgg19.w24.bias.bin", "rb");
	size_t model_matrix_words = fread(model_matrix, sizeof(float), model_dim_in*model_dim_out, f_model_matrix);
	size_t model_bias_words = fread(model_bias, sizeof(float), model_dim_out, f_model_bias);
	if ( model_dim_in*model_dim_out != model_matrix_words ) {
		fprintf(stderr, "error: Model matrix size mismatch!\n" );
		exit(1);
	}
	if ( model_dim_out != model_bias_words ) {
		fprintf(stderr, "error: Model bias size mismatch!\n" );
		exit(1);
	}


	printf( "Model loaded successfuly!\n" );

	size_t input_cnt = 0;
	float* inputs = (float*)aligned_alloc(64,sizeof(float)*model_dim_in*maximum_inputs);
	float* outputs = (float*)aligned_alloc(64,sizeof(float)*model_dim_out*maximum_inputs);
	float* outputs_golden = (float*)malloc(sizeof(float)*model_dim_out*maximum_inputs);
	FILE* f_input = fopen(fname_input, "rb");
	FILE* f_output = fopen(fname_output, "rb");
	while(true) {
		size_t input_words = fread(inputs+input_cnt*model_dim_in, sizeof(float), model_dim_in, f_input);
		size_t output_words = fread(outputs_golden+input_cnt*model_dim_out, sizeof(float), model_dim_out, f_output);
		if ( input_words != model_dim_in || output_words != model_dim_out ) break;
		input_cnt++;
	}
	if ( input_cnt < 1 ) {
		fprintf( stderr, "error: No example data loaded\n" );
		exit(2);
	}

	printf( "Example data loaded successfuly!\nInput count: %ld\nStarting layer...\n", input_cnt );

	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point now;
	std::chrono::microseconds duration_micro;

	start = std::chrono::high_resolution_clock::now();
	fc_layer(input_cnt, model_dim_in, model_dim_out, model_matrix, model_bias, inputs, outputs, threads);
	now = std::chrono::high_resolution_clock::now();
	duration_micro = std::chrono::duration_cast<std::chrono::microseconds> (now-start);
	printf( "Elapsed time: %f s\n", 0.000001f*duration_micro.count() );
	size_t total_flop = input_cnt*2*((model_dim_in+1)*model_dim_out);
	size_t flopus = total_flop/duration_micro.count();
	double mflop = (double)flopus;
	printf( "Performance (MFLOPS): %lf\n", mflop );


	double totalnoise = 0;
	for (size_t oidx = 0; oidx < input_cnt; oidx++) {
		size_t origin = oidx*model_dim_out;
		float noise = 0;
		for ( size_t i = origin; i < origin+model_dim_out; i++ ) {
			noise += std::abs(outputs[i]-outputs_golden[i]);
		}
		totalnoise += noise;
		if ( noise > 1 ) {
			printf( "Warning: error larger than 1 for input %ld : %f\n", oidx, noise );
		}
	}
	printf( "Average error: %lf\n", totalnoise/input_cnt );

	return 0;
}
