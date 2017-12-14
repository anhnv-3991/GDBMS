#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include "gpuij.h"
#include <sys/time.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <inttypes.h>
#include <thrust/system/cuda/execution_policy.h>
#include "utilities.h"

namespace voltdb {

GPUIJ::GPUIJ()
{
		join_result_ = NULL;
		result_size_ = 0;
		lookup_type_ = INDEX_LOOKUP_TYPE_EQ;
}

GPUIJ::GPUIJ(GTable outer_table,
				GTable inner_table,
				std::vector<ExpressionNode*> search_exp,
				ExpressionNode *end_expression,
				ExpressionNode *post_expression,
				ExpressionNode *initial_expression,
				ExpressionNode *skipNullExpr,
				ExpressionNode *prejoin_expression,
				ExpressionNode *where_expression,
				IndexLookupType lookup_type)
{
	/**** Table data *********/
	outer_table_ = outer_table;
	inner_table_ = inner_table;
	join_result_ = NULL;
	result_size_ = 0;
	lookup_type_ = lookup_type;

	/**** Expression data ****/
	search_exp_ = GExpressionVector(search_exp);

	end_expression_ = GExpression(end_expression);

	post_expression_ = GExpression(post_expression);

	initial_expression_ = GExpression(initial_expression);

	skipNullExpr_ = GExpression(skipNullExpr);

	prejoin_expression_ = GExpression(prejoin_expression);

	where_expression_ = GExpression(where_expression);
}

GPUIJ::~GPUIJ()
{
	free(join_result_);
	search_exp_.free();
	end_expression_.free();
	post_expression_.free();
	initial_expression_.free();
	skipNullExpr_.free();
	prejoin_expression_.free();
	where_expression_.free();
}

bool GPUIJ::execute(){
	gettimeofday(&all_start_, NULL);

	/******** Calculate size of blocks, grids, and GPU buffers *********/
	uint gpu_size = 0, part_size = 0;
	ulong jr_size;

	RESULT *jresult_dev, *write_dev;
	jresult_dev = write_dev = NULL;
	ulong *index_psum, *exp_psum;
	ResBound *res_bound;
	bool *prejoin_res_dev;

	part_size = getPartitionSize();

//	int block_x, grid_x;
//
//	block_x = (part_size < BLOCK_SIZE_X) ? part_size : BLOCK_SIZE_X;
//	grid_x = (part_size - 1)/block_x + 1;
	gpu_size = DEFAULT_PART_SIZE_ + 1;

	/******** Allocate GPU buffer for table data and counting data *****/
	checkCudaErrors(cudaMalloc(&prejoin_res_dev, part_size * sizeof(bool)));
	checkCudaErrors(cudaMalloc(&index_psum, gpu_size * sizeof(ulong)));

	checkCudaErrors(cudaMalloc(&res_bound, gpu_size * sizeof(ResBound)));

	struct timeval pre_start, pre_end, istart, iend, estart, eend, pestart, peend, wstart, wend, end_join, balance_start, balance_end;

	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (int outer_idx = 0; outer_idx < outer_table_.getBlockNum(); outer_idx++) {
		//Size of outer small table
		outer_table_.moveToBlock(outer_idx);
		gpu_size = outer_table_.getCurrentRowNum() + 1;

		/* Evaluate prejoin predicate */
		gettimeofday(&pre_start, NULL);
		PrejoinFilter(prejoin_res_dev);
		gettimeofday(&pre_end, NULL);
		prejoin_.push_back(timeDiff(pre_start, pre_end));

		joins_only_.push_back(timeDiff(pre_start, pre_end));

		for (int inner_idx = 0; inner_idx < inner_table_.getBlockNum(); inner_idx++) {
			/* Binary search for index */
			inner_table_.moveToBlock(inner_idx);
			gettimeofday(&istart, NULL);

			indexFilter(index_psum, res_bound, prejoin_res_dev);

			gettimeofday(&iend, NULL);
			index_.push_back(timeDiff(istart, iend));

			RESULT *tmp_result;
			ulong tmp_size = 0;

			gettimeofday(&balance_start, NULL);
			rebalance(index_psum, res_bound, &tmp_result, gpu_size, &tmp_size);
			gettimeofday(&balance_end, NULL);

			rebalance_.push_back(timeDiff(balance_start, balance_end));

			if (tmp_size == 0) {
				gettimeofday(&end_join, NULL);
				joins_only_.push_back(timeDiff(istart, end_join));
				continue;
			}
			checkCudaErrors(cudaMalloc(&jresult_dev, tmp_size * sizeof(RESULT)));
			checkCudaErrors(cudaMalloc(&exp_psum, (tmp_size + 1) * sizeof(ulong)));

			gettimeofday(&estart, NULL);
			expressionFilter(tmp_result, jresult_dev, exp_psum, tmp_size);
			gettimeofday(&eend, NULL);

			expression_.push_back(timeDiff(estart, eend));

			gettimeofday(&pestart, NULL);
			GUtilities::ExclusiveScan(exp_psum, tmp_size + 1, &jr_size);
			gettimeofday(&peend, NULL);

			epsum_.push_back(timeDiff(pestart, peend));

			checkCudaErrors(cudaFree(tmp_result));

			if (jr_size == 0) {
				continue;
			}
			checkCudaErrors(cudaMalloc(&write_dev, jr_size * sizeof(RESULT)));

			gettimeofday(&wstart, NULL);
			GUtilities::removeEmptyResult(write_dev, jresult_dev, exp_psum, tmp_size);
			gettimeofday(&wend, NULL);
			wtime_.push_back(timeDiff(wstart, wend));

			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size) * sizeof(RESULT));

			gettimeofday(&end_join, NULL);
			checkCudaErrors(cudaMemcpy(join_result_ + result_size_, write_dev, jr_size * sizeof(RESULT), cudaMemcpyDeviceToHost));

			result_size_ += jr_size;
			jr_size = 0;

			joins_only_.push_back(timeDiff(istart, end_join));
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());
	/******** Free GPU memory, unload module, end session **************/

	checkCudaErrors(cudaFree(res_bound));
	checkCudaErrors(cudaFree(prejoin_res_dev));
	gettimeofday(&all_end_, NULL);

	return true;
}

void GPUIJ::getResult(RESULT *output) const
{
	memcpy(output, join_result_, sizeof(RESULT) * result_size_);
}

int GPUIJ::getResultSize() const
{
	return result_size_;
}

uint GPUIJ::getPartitionSize() const
{
	int part_size = DEFAULT_PART_SIZE_;
	int outer_size = outer_table_.getCurrentRowNum();
	int inner_size = inner_table_.getCurrentRowNum();
	int bigger_tuple_size = (outer_size > inner_size) ? outer_size : inner_size;

	if (bigger_tuple_size < part_size) {
		return bigger_tuple_size;
	}

	for (uint i = 32768; i <= DEFAULT_PART_SIZE_; i = i * 2) {
		if (bigger_tuple_size < i) {
			part_size = i;
			break;
		}
	}

	printf("getPartitionSize: PART SIZE = %d\n", part_size);
	return part_size;
}


void GPUIJ::debug(void)
{
	printf("******** Debugging information *********** \n");
	printf("EXPRESSIONS:\n");

	printf("End Expression: ");
	end_expression_.debug();

	printf("Post Expression: ");
	post_expression_.debug();

	printf("Initial Expression: ");
	initial_expression_.debug();

	printf("Skip Null Expression: ");
	skipNullExpr_.debug();

	printf("Where Expression: ");
	where_expression_.debug();

	printf("\nTABLES:\n");
	printf("Outer table:");
	outer_table_.debug();

	printf("Inner table:");
	inner_table_.debug();
}


void GPUIJ::profiling()
{
	unsigned long allocation_time = 0, prejoin_time = 0, index_time = 0, expression_time = 0, ipsum_time = 0, epsum_time = 0, wtime_time = 0, joins_only_time = 0, all_time = 0;

	for (int i = 0; i < prejoin_.size(); i++) {
		prejoin_time += prejoin_[i];
	}

	for (int i = 0; i < index_.size(); i++) {
		index_time += index_[i];
	}

	for (int i = 0; i < expression_.size(); i++) {
		expression_time += expression_[i];
	}

	for (int i = 0; i < ipsum_.size(); i++) {
		ipsum_time += ipsum_[i];
	}

	for (int i = 0; i < epsum_.size(); i++) {
		epsum_time += epsum_[i];
	}

	for (int i = 0; i < wtime_.size(); i++) {
		wtime_time += wtime_[i];
	}

#if (defined(DECOMPOSED1_) || defined(DECOMPOSED2_))
	unsigned long rebalance_cost = 0;
	for (int i = 0; i < rebalance_.size(); i++) {
		rebalance_cost += rebalance_[i];
	}
#endif


	for (int i = 0; i < joins_only_.size(); i++) {
		joins_only_time += joins_only_[i];
	}

	all_time = (all_end_.tv_sec - all_start_.tv_sec) * 1000000 + (all_end_.tv_usec - all_start_.tv_usec);

	allocation_time = all_time - joins_only_time;
	printf("**********************************\n"
			"Allocation & data movement time: %lu\n"
			"Prejoin filter Time: %lu\n"
			"Index Search Time: %lu\n"
			"Rebalance Cost: %lu\n"
			"Expression filter Time: %lu\n"
			"Expression Prefix Sum Time: %lu\n"
			"Write back time Time: %lu\n"
			"Joins Only Time: %lu\n"
			"Total join time: %lu\n"
			"*******************************\n",
			allocation_time, prejoin_time, index_time,
			rebalance_cost, expression_time, epsum_time, wtime_time, joins_only_time, all_time);

}

unsigned long GPUIJ::timeDiff(struct timeval start, struct timeval end)
{
	return GUtilities::timeDiff(start, end);
}

__global__ void prejoinDev(GTable outer, int outer_rows, GExpression prejoin, bool *result,GNValue *stack)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	GStack tmp_stk(stack + index, offset);

	GTuple outer_tuple;

	for (int i = index; i < outer_rows; i+= offset) {
		GNValue res = GNValue::getTrue();
		outer_tuple = outer.getGTuple(i);

		if (prejoin.size() > 1) {
			res = prejoin.evaluate(outer_tuple, GTuple(), tmp_stk);
		}

		result[i] = res.isTrue();
	}
}

void GPUIJ::prejoinFilter(bool *result)
{
	int outer_rows = outer_table_.getCurrentRowNum();
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

	int stack_size = prejoin_expression_.height();
	GNValue *stack = NULL;

	if (stack_size > 0) {
		checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * stack_size));
	}

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	prejoinDev<<<grid_size, block_size>>>(outer_table_, outer_rows, prejoin_expression_, result, stack);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if (stack_size > 0) {
		checkCudaErrors(cudaFree(stack));
	}
}

void GPUIJ::prejoinFilter(bool *result, cudaStream_t stream)
{
	int outer_rows = outer_table_.getCurrentRowNum();
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

	int stack_size = prejoin_expression_.height();
	GNValue *stack = NULL;

	if (stack_size > 0) {
		checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * stack_size));
	}

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	prejoinDev<<<grid_size, block_size, 0, stream>>>(outer_table_, outer_rows, prejoin_expression_, result, stack);
	checkCudaErrors(cudaGetLastError());

	if (stack_size > 0) {
		checkCudaErrors(cudaFree(stack));
	}
}

__global__ void decomposeDev(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		out[i].lkey = in[in_location[i]].outer;
		out[i].rkey = in[in_location[i]].left + local_offset[i];
	}
}

void GPUIJ::decompose(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	decomposeDev<<<grid_size, block_x>>>(in, out, in_location, local_offset, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GPUIJ::decompose(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	decomposeDev<<<grid_size, block_x, 0, stream>>>(in, out, in_location, local_offset, size);
	checkCudaErrors(cudaGetLastError());
}


__global__ void lowerBound(GTable search_table, GTreeIndex inner_idx,
							int search_rows, int inner_rows,
							ulong *index_psum, ResBound *res_bound,
							IndexLookupType lookup_type,
							bool *prejoin_res_dev
						  )

{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	GTreeIndexKey outer_key;

	for (int i = index; i < search_rows; i += offset) {
		res_bound[i].left = -1;
		res_bound[i].outer = -1;

		if (prejoin_res_dev[i]) {
			res_bound[i].outer = i;

			GTuple tuple = search_table.getGTuple(i);
			outer_key.createKey(tuple);

			switch (lookup_type) {
				case INDEX_LOOKUP_TYPE_EQ:
				case INDEX_LOOKUP_TYPE_GT:
				case INDEX_LOOKUP_TYPE_GTE:
				case INDEX_LOOKUP_TYPE_LT: {
					res_bound[i].left = inner_idx.lowerBound(outer_key);
					break;
				}
				case INDEX_LOOKUP_TYPE_LTE: {
					res_bound[i].left = 0;
					break;
				}
				default:
					break;
			}
		}
	}
}

__global__ void upperBound(GTable search_table, GTreeIndex inner_idx,
								int search_rows, int inner_rows,
								ulong *index_psum, ResBound *res_bound,
								IndexLookupType lookup_type,
								bool *prejoin_res_dev
							  )

{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	GTreeIndexKey outer_key;

	for (int i = index; i < search_rows; i += offset) {
		index_psum[i] = 0;
		res_bound[i].right = -1;

		if (prejoin_res_dev[i]) {
			GTuple tuple = search_table.getGTuple(i);
			outer_key.createKey(tuple);

			switch (lookup_type) {
			case INDEX_LOOKUP_TYPE_EQ:
			case INDEX_LOOKUP_TYPE_LTE: {
				res_bound[i].right = inner_idx.upperBound(outer_key);
				break;
			}
			case INDEX_LOOKUP_TYPE_GT:
			case INDEX_LOOKUP_TYPE_GTE: {
				res_bound[i].right = inner_rows;
				break;
			}
			case INDEX_LOOKUP_TYPE_LT: {
				res_bound[i].right = res_bound[i].left - 1;
				res_bound[i].left = 0;
				break;
			}
			default:
				break;
			}
		}

		index_psum[i] = (res_bound[i].right >= 0 && res_bound[i].left >= 0) ? (res_bound[i].right - res_bound[i].left + 1) : 0;
	}

	if (index == 0)
		index_psum[search_rows] = 0;
}


__global__ void constructSearchTable(GTable outer_table, GTable search_table,
										int outer_rows,
										GExpressionVector search_exp,
										GNValue *stack)
{
	GNValue tmp;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTuple tuple;
	GStack tmp_stk(stack + index, stride);

	for (int i = index; i < outer_rows; i += stride) {
		tuple = search_table.getGTuple(i);

		for (int j = 0; j < search_exp.size(); j++) {
			tmp_stk.reset();
			tmp = search_exp.at(j).evaluate(tuple, GTuple(), tmp_stk);
			tuple.setGNValue(tmp, j);
		}
	}
}

void GPUIJ::indexFilter(ulong *index_psum, ResBound *res_bound, bool *prejoin_res_dev)
{
	int outer_rows = outer_table_.getCurrentRowNum(), inner_rows = inner_table_.getCurrentRowNum();
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

	int stack_size = 0;

	for (int i = 0; i < search_exp_.size(); i++) {
		if (search_exp_[i].height() > stack_size)
			stack_size = search_exp_[i].height();
	}

	GNValue *stack;

	if (stack_size > 0) {
		checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * stack_size));
	}

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);
	GTable search_table(NULL, search_exp_.size());
	GTreeIndex *inner_idx = static_cast<GTreeIndex*>(inner_table_.getCurrentIndex());

	constructSearchTable<<<grid_size, block_size>>>(outer_table_, search_table, outer_rows, search_exp_, stack);
	checkCudaErrors(cudaGetLastError());

	lowerBound<<<grid_size, block_size>>>(search_table, *inner_idx, outer_rows, inner_rows, index_psum, res_bound, lookup_type_, prejoin_res_dev);
	checkCudaErrors(cudaGetLastError());

	upperBound<<<grid_size, block_size>>>(search_table, *inner_idx, outer_rows, inner_rows, index_psum, res_bound, lookup_type_, prejoin_res_dev);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());

	search_table.removeTable();

	if (stack_size > 0) {
		checkCudaErrors(cudaFree(stack));
	}

}

void GPUIJ::indexFilter(ulong *index_psum, ResBound *res_bound, bool *prejoin_res_dev, cudaStream_t stream)
{
	int outer_rows = outer_table_.getCurrentRowNum(), inner_rows = inner_table_.getCurrentRowNum();
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1) / block_x + 1;

	int stack_size = 0;

	for (int i = 0; i < search_exp_.size(); i++) {
		if (search_exp_[i].height() > stack_size)
			stack_size = search_exp_[i].height();
	}

	GNValue *stack;

	if (stack_size > 0) {
		checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * stack_size));
	}

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);
	GTable search_table(NULL, search_exp_.size());
	GTreeIndex *inner_idx = static_cast<GTreeIndex*>(inner_table_.getCurrentIndex());

	constructSearchTable<<<grid_size, block_size, 0, stream>>>(outer_table_, search_table, outer_rows, search_exp_, stack);
	checkCudaErrors(cudaGetLastError());

	lowerBound<<<grid_size, block_size, 0, stream>>>(search_table, *inner_idx, outer_rows, inner_rows, index_psum, res_bound, lookup_type_, prejoin_res_dev);
	checkCudaErrors(cudaGetLastError());

	upperBound<<<grid_size, block_size, 0, stream>>>(search_table, *inner_idx, outer_rows, inner_rows, index_psum, res_bound, lookup_type_, prejoin_res_dev);
	checkCudaErrors(cudaGetLastError());

	search_table.removeTable();

	if (stack_size > 0) {
		checkCudaErrors(cudaFree(stack));
	}
}


__global__ void expFilter(GTable outer, GTable inner,
							RESULT *in_bound, RESULT *out_bound,
							ulong *mark_location, int size,
							GExpression end_exp, GExpression post_exp, GExpression where_exp,
							GNValue *stack)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	GNValue res;
	GTuple outer_tuple, inner_tuple;
	GStack tmp_stk(stack + index, offset);

	for (int i = index; i < size; i += offset) {
		res = GNValue::getTrue();
		outer_tuple = outer.getGTuple(in_bound[i].lkey);
		inner_tuple = inner.getGTuple(in_bound[i].rkey);
		tmp_stk.reset();

		if (end_exp.size() > 0) {
			res = end_exp.evaluate(outer_tuple, inner_tuple, tmp_stk);
		}

		if (post_exp.size() > 0) {
			res = post_exp.evaluate(outer_tuple, inner_tuple, tmp_stk) && res;
		}

		if (where_exp.size() > 0) {
			res = where_exp.evaluate(outer_tuple, inner_tuple, tmp_stk) && res;
		}

		out_bound[i].lkey = (res.isTrue()) ? in_bound[i].lkey : (-1);
		out_bound[i].rkey = (res.isTrue()) ? in_bound[i].rkey : (-1);
		mark_location[i] = (res.isTrue()) ? 1 : 0;
	}

	if (index == 0) {
		mark_location[size] = 0;
	}
}

void GPUIJ::expressionFilter(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size)
{
	int partition_size = DEFAULT_PART_SIZE_;

	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size <= partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

	int stack_size = end_expression_.height();

	if (stack_size < post_expression_.height()) {
		stack_size = post_expression_.height();
	}

	if (stack_size < where_expression_.height()) {
		stack_size = post_expression_.height();
	}

	GNValue *stack = NULL;

	if (stack_size > 0) {
		checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * stack_size));
	}

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	expFilter<<<grid_size, block_size>>>(outer_table_, inner_table_,
											in_bound, out_bound,
											mark_location, size,
											end_expression_, post_expression_, where_expression_,
											stack);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if (stack_size > 0) {
		checkCudaErrors(cudaFree(stack));
	}
}

void GPUIJ::expressionFilter(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size, cudaStream_t stream)
{
	int partition_size = DEFAULT_PART_SIZE_;

	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size <= partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

	int stack_size = end_expression_.height();

	if (stack_size < post_expression_.height()) {
		stack_size = post_expression_.height();
	}

	if (stack_size < where_expression_.height()) {
		stack_size = post_expression_.height();
	}

	GNValue *stack = NULL;

	if (stack_size > 0) {
		checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * stack_size));
	}

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	expFilter<<<grid_size, block_size, 0, stream>>>(outer_table_, inner_table_,
														in_bound, out_bound,
														mark_location, size,
														end_expression_, post_expression_, where_expression_,
														stack);
	checkCudaErrors(cudaGetLastError());

	if (stack_size > 0) {
		checkCudaErrors(cudaFree(stack));
	}
}

__global__ void expFilter2(GTable outer, GTable inner,
							int outer_rows,
							RESULT *result, ulong *index_psum,
							ulong *exp_psum, uint result_size,
							GExpression end_dev, GExpression post_dev, GExpression where_dev,
							ResBound *res_bound, bool *prejoin_res_dev,
							GNValue *stack)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	GTuple outer_tuple, inner_tuple;
	GStack tmp_stk(stack + index, offset);

	for (int i = index; i < outer_rows; i += offset) {
		exp_psum[i] = 0;
		ulong writeloc = index_psum[index];
		int count = 0;
		int res_left = -1, res_right = -1;
		GNValue res = GNValue::getTrue();

		res_left = res_bound[i].left;
		res_right = res_bound[i].right;

		tmp_stk.reset();

		while (res_left >= 0 && res_left <= res_right && writeloc < result_size) {
			outer_tuple = outer.getGTuple(res_left);
			inner_tuple = inner.getGTuple(res_right);

			if (end_dev.size() > 0) {
				res = end_dev.evaluate(outer_tuple, inner_tuple, tmp_stk);
			}

			if (post_dev.size() > 0) {
				res = end_dev.evaluate(outer_tuple, inner_tuple, tmp_stk) && res;
			}

			if (where_dev.size() > 0) {
				res = where_dev.evaluate(outer_tuple, inner_tuple, tmp_stk) && res;
			}

			result[writeloc].lkey = (res.isTrue()) ? i : (-1);
			result[writeloc].rkey = (res.isTrue()) ? res_left : (-1);
			count += (res.isTrue()) ? 1 : 0;
			writeloc++;
			res_left++;
		}
		exp_psum[i] = count;
	}

	if (index == 0) {
		exp_psum[outer_rows] = 0;
	}
}

void GPUIJ::expressionFilter(ulong *index_psum, ulong *exp_psum, RESULT *result, int result_size, ResBound *res_bound, bool *prejoin_res_dev)
{
	int outer_rows = outer_table_.getCurrentRowNum();
	int partition_size = DEFAULT_PART_SIZE_;
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows < partition_size) ? (outer_rows - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

	int stack_size = end_expression_.height();

	if (stack_size < post_expression_.height()) {
		stack_size = post_expression_.height();
	}

	if (stack_size < where_expression_.height()) {
		stack_size = post_expression_.height();
	}

	GNValue *stack = NULL;

	if (stack_size > 0) {
		checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * stack_size));
	}

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	expFilter2<<<grid_size, block_size>>>(outer_table_, inner_table_,
													outer_rows,
													result, index_psum,
													exp_psum,
													result_size,
													end_expression_, post_expression_, where_expression_,
													res_bound, prejoin_res_dev, stack
													);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if (stack_size > 0) {
		checkCudaErrors(cudaFree(stack));
	}
}

void GPUIJ::expressionFilter(ulong *index_psum, ulong *exp_psum, RESULT *result, int result_size, ResBound *res_bound, bool *prejoin_res_dev, cudaStream_t stream)
{
	int outer_rows = outer_table_.getCurrentRowNum();
	int partition_size = DEFAULT_PART_SIZE_;
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows < partition_size) ? (outer_rows - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

	int stack_size = end_expression_.height();

	if (stack_size < post_expression_.height()) {
		stack_size = post_expression_.height();
	}

	if (stack_size < where_expression_.height()) {
		stack_size = post_expression_.height();
	}

	GNValue *stack = NULL;

	if (stack_size > 0) {
		checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * stack_size));
	}

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	expFilter2<<<grid_size, block_size, 0, stream>>>(outer_table_, inner_table_,
														outer_rows,
														result, index_psum,
														exp_psum, result_size,
														end_expression_, post_expression_, where_expression_,
														res_bound, prejoin_res_dev, stack);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));

	if (stack_size > 0) {
		checkCudaErrors(cudaFree(stack));
	}
}

void GPUIJ::rebalance(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
{
	GUtilities::ExclusiveScan(in, in_size, out_size, stream);

	if (*out_size == 0) {
		return;
	}

	ulong *location;

	checkCudaErrors(cudaMalloc(&location, sizeof(ulong) * (*out_size)));

	checkCudaErrors(cudaMemsetAsync(location, 0, sizeof(ulong) * (*out_size), stream));

	GUtilities::markLocation(location, in, in_size, stream);

	GUtilities::InclusiveScan(location, *out_size, stream);

	ulong *local_offset;

	checkCudaErrors(cudaMalloc(&local_offset, *out_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, *out_size * sizeof(RESULT)));

	GUtilities::computeOffset(in, location, local_offset, *out_size, stream);

	decompose(in_bound, *out_bound, location, local_offset, *out_size, stream);

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(location));
}

void GPUIJ::rebalance(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size)
{
	GUtilities::ExclusiveScan(in, in_size, out_size);

	if (*out_size == 0) {
		return;
	}

	ulong *location;

	checkCudaErrors(cudaMalloc(&location, sizeof(ulong) * (*out_size)));

	checkCudaErrors(cudaMemset(location, 0, sizeof(ulong) * (*out_size)));

	checkCudaErrors(cudaDeviceSynchronize());

	GUtilities::markLocation(location, in, in_size);

	GUtilities::inclusiveScan(location, *out_size);

	ulong *local_offset;

	checkCudaErrors(cudaMalloc(&local_offset, *out_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, *out_size * sizeof(RESULT)));

	GUtilities::computeOffset(in, location, local_offset, *out_size);

	decompose(in_bound, *out_bound, location, local_offset, *out_size);

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(location));
}
}
