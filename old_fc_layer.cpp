#include <stdio.h>

void
fc_layer(size_t data_cnt, size_t input_dim, size_t output_dim, float* matrix, float* bias, float* input, float* output, int threads) {
	// loop over input instances
	for ( size_t iidx = 0; iidx < data_cnt; iidx++ ) {
		// loop over weight columns
		for ( size_t oidx = 0; oidx < output_dim; oidx++ ) {
			float outv = 0;
			// loop over each input's activation values
			for ( size_t aidx = 0; aidx < input_dim; aidx++ ) {
				float inv = input[input_dim*iidx+aidx];
				float weight = matrix[output_dim*aidx+oidx];
				outv += inv*weight;
			}
			outv += bias[oidx];
			if ( outv < 0 ) outv = 0;

			output[iidx*output_dim+oidx] = outv;
		}
	}
}
