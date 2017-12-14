#ifndef GPU_COMMON_H_
#define GPU_COMMON_H_

#include <iostream>
#include <stdint.h>
#include "macros.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace voltdb {

class GUtilities {
public:
	static void removeEmptyResult(RESULT *out_bound, RESULT *in_bound, ulong *in_location, ulong *out_location, uint in_size);
	static void removeEmptyResult(RESULT *out_bound, RESULT *in_bound, ulong *in_location, ulong *out_location, uint in_size, cudaStream_t stream);

	static void removeEmptyResult(RESULT *out, RESULT *in, ulong *location, int size);
	static void removeEmptyResult(RESULT *out, RESULT *in, ulong *location, int size, cudaStream_t stream);

	static void markNonZeros(ulong *input, int size, ulong *output);
	static void markNonZeros(ulong *input, int size, ulong *output, cudaStream_t stream);

	static void removeZeros(ulong *input, ResBound *in_bound, ulong *output, ResBound *out_bound, ulong *output_location, int size);
	static void removeZeros(ulong *input, ResBound *in_bound, ulong *output, ResBound *out_bound, ulong *output_location, int size, cudaStream_t stream);

	static void markTmpLocation(ulong *tmp_location, ulong *input, int size);
	static void markTmpLocation(ulong *tmp_location, ulong *input, int size, cudaStream_t stream);

	static void markLocation(ulong *location, ulong *input, int size);
	static void markLocation(ulong *location, ulong *input, int size, cudaStream_t stream);

	static void computeOffset(ulong *input1, ulong *input2, ulong *out, int size);
	static void computeOffset(ulong *input1, ulong *input2, ulong *out, int size, cudaStream_t stream);

	template <typename T = ulong> static void ExclusiveScan(T *input, int ele_num, T *sum);
	template <typename T = ulong> static void ExclusiveScan(T *input, int ele_num, T *sum, cudaStream_t stream);
	template <typename T = ulong> static void ExclusiveScan(T *input, int ele_num);
	template <typename T = ulong> static void ExclusiveScan(T *input, int ele_num, cudaStream_t stream);


	template <typename T = ulong> static void InclusiveScan(T *input, int ele_num);
	template <typename T = ulong> static void InclusiveScan(T *input, int ele_num, cudaStream_t stream);

	static unsigned long timeDiff(struct timeval start, struct timeval end);
};


}
#endif
