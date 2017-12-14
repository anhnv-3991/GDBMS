#include "projection.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include <sstream>

namespace voltdb {
GExecutorProjection::GExecutorProjection()
{
	output_ = NULL;
	tuple_array_ = NULL;
	param_array_ = NULL;
	param_ = NULL;
}

GExecutorProjection::GExecutorProjection(GTable *output_table, GTable input_table, int *tuple_array, int *param_array, GNValue *param, std::vector<ExpressionNode *> expression)
{
	output_ = output_table;
	input_ = input_table;

	int columns = output_->getColumnCount();

	if (tuple_array != NULL) {
		checkCudaErrors(cudaMalloc(&tuple_array_, sizeof(int) * columns));
		checkCudaErrors(cudaMemcpy(tuple_array_, tuple_array, sizeof(int) * columns, cudaMemcpyHostToDevice));
	} else
		tuple_array_ = NULL;

	if (param_array != NULL) {
		checkCudaErrors(cudaMalloc(&param_array_, sizeof(int) * columns));
		checkCudaErrors(cudaMemcpy(param_array_, param_array, sizeof(int) * columns, cudaMemcpyHostToDevice));
	} else
		param_array_ = NULL;

	if (param != NULL) {
		checkCudaErrors(cudaMalloc(&param_, sizeof(GNValue) * columns));
		checkCudaErrors(cudaMemcpy(param_, param, sizeof(GNValue) * columns, cudaMemcpyHostToDevice));
	} else
		param_ = NULL;

	expression_ = GExpressionVector(expression);
}

bool GExecutorProjection::execute()
{
	int rows;

	for (int i = 0; i < input_.getBlockNum(); i++) {
		input_.moveToBlock(i);
		output_->addBlock();
		rows = input_.getCurrentRowNum();
		output_->setBlockRow(i, rows);
		output_->setTupleCount(output_->getColumnCount() + rows);

		evaluate();
	}

	return true;
}

__global__ void evaluate0(GTable output, GTable input, int *tuple_array, int rows)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTuple input_tuple, output_tuple;
	GNValue tmp;
	int columns = output.getColumnCount();

	for (int i = index; i < rows; i += stride) {
		input_tuple = input.getGTuple(i);
		output_tuple = output.getGTuple(i);

		for (int j = 0; j < columns; j++) {
			tmp = input_tuple.getGNValue(tuple_array[j]);
			output_tuple.setGNValue(tmp, j);
		}
	}
}

__global__ void evaluate1(GTable output, int *param_array, GNValue *param, int rows)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTuple output_tuple;
	GNValue tmp;
	int columns = output.getColumnCount();

	for (int i = index; i < rows; i += stride) {
		output_tuple = output.getGTuple(i);

		for (int j = 0; j < columns; j++)
			output_tuple.setGNValue(param[param_array[j]], j);
	}
}

__global__ void evaluate2(GTable output, GTable input, GExpressionVector expression, int rows, GNValue *stack)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTuple input_tuple, output_tuple;
	GNValue tmp;
	GTuple dummy;
	int columns = output.getColumnCount();
	GStack tmp_stk(stack + index, stride);

	for (int i = index; i < rows; i += stride) {
		output_tuple = output.getGTuple(i);
		input_tuple = input.getGTuple(i);

		for (int j = 0; j < expression.size(); j++) {
			tmp_stk.reset();
			tmp = expression.at(j).evaluate(input_tuple, dummy, tmp_stk);

			output_tuple.setGNValue(tmp, j);
		}
	}
}

void GExecutorProjection::evaluate()
{
	int rows = input_.getCurrentRowNum();
	int block_x = (rows > BLOCK_SIZE_X) ? BLOCK_SIZE_X : rows;
	int grid_x = (rows - 1) / block_x + 1;



	if (tuple_array_ != NULL) {
		evaluate0<<<grid_x, block_x>>>(*output_, input_, tuple_array_, rows);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

	} else if (param_array_ != NULL) {
		evaluate1<<<grid_x, block_x>>>(*output_, param_array_, param_, rows);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	} else {

		int stack_size = 0;
		GNValue *stack = NULL;

		for (int i = 0; i < expression_.size(); i++) {
			if (stack_size < expression_[i].height())
				stack_size = expression_[i].height();
		}

		checkCudaErrors(cudaMalloc(stack, sizeof(GNValue) * grid_x * block_x * stack_size));

		evaluate2<<<grid_x, block_x>>>(*output_, input_, expression_, rows, stack);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		if (stack_size > 0) {
			checkCudaErrors(cudaFree(stack));
		}
	}
}

std::string GExecutorProjection::debug() const
{
	std::ostringstream output;

	output << "DEBUG Type: Projection Executor" << std::endl;
	output << "Input table:" << std::endl;
	output << input_.debug() << std::endl;
	output << "Output table:" << std::endl;
	output << output_->debug() << std::endl;
	output << "Tuple array:" << std::endl;

	int columns = input_.getColumnCount();

	if (columns > 0 && tuple_array_ != NULL) {
		for (int i = 0; i < columns; i++) {
			output << "Tuple: " << tuple_array_[i];
			if (i < columns - 1)
				output << "::";
		}
		output << std::endl;
	} else
		output << "Empty" << std::endl;

	output << "Param list:" << std::endl;
	if (columns > 0 && param_array_ != NULL && param_ != NULL) {

		for (int i = 0; i < columns; i++) {
			output << "[" << param_array_[i] << "]:" << param_[param_array_[i]].debug();
			if (i < columns - 1)
				output << "::";
		}
		output << std::endl;

	} else
		output << "Empty" << std::endl;

	output << expression_.debug() << std::endl;

	std::string retval(output.str());

	return retval;
}

GExecutorProjection::~GExecutorProjection()
{
	if (tuple_array_ != NULL) {
		checkCudaErrors(cudaFree(tuple_array_));
		tuple_array_ = NULL;
	}

	if (param_array_ != NULL) {
		checkCudaErrors(cudaFree(param_array_));
		param_array_ = NULL;
	}

	if (param_ != NULL) {
		checkCudaErrors(cudaFree(param_));
		param_ = NULL;
	}

	expression_.free();
}
}
