#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "gexpression.h"
#include <string>
#include <sstream>

namespace voltdb {

GExpression::GExpression(ExpressionNode *expression) {
	int size = 0;

	size =	getExpressionLength(expression);

	size_ = size;

	if (size_ > 0) {
		checkCudaErrors(cudaMalloc(&expression_, tree_size * sizeof(GTreeNode)));
		createExpression(expression);
	} else {
		size_ = 0;
		expression_ = NULL;
	}
}

bool GExpression::createExpression(ExpressionNode *expression) {
	GTreeNode *tmp_expression = (GTreeNode*)malloc(sizeof(GTreeNode) * size_);

	int root = 0;

	if (!buildPostExpression(tmp_expression, expression, &root))
		return false;

	checkCudaErrors(cudaMemcpy(expression_, tmp_expression, sizeof(GTreeNode) * size_, cudaMemcpyHostToDevice));
	std::free(tmp_expression);

	return true;
}

void GExpression::free() {
	if (size_ > 0) {
		checkCudaErrors(cudaFree(expression_));
		expression_ = NULL;
		size_ = 0;
	}
}

int GExpression::getExpressionLength(ExpressionNode *expression) {
	if (expression == NULL) {
		return 0;
	}

	int left, right;

	left = getExpressionLength(expression->left);
	right = getExpressionLength(expression->right);

	return (1 + left + right);
}

bool GExpression::buildPostExpression(GTreeNode *output_expression, ExpressionNode *expression, int *index) {
	if (expression == NULL)
		return true;

	if (size_ <= *index)
		return false;

	if (!buildPostExpression(output_expression, expression->left, index))
		return false;

	if (!buildPostExpression(output_expression, expression->right, index))
		return false;

	output_expression[*index] = expression->node;
	(*index)++;

	return true;
}

std::string GExpression::debug() const
{
	if (size_ == 0) {
		std::string retval("Empty expression");
		return retval;
	}

	std::ostringstream output;

	GTreeNode *expression_host = (GTreeNode*)malloc(sizeof(GTreeNode) * size_);

	checkCudaErrors(cudaMemcpy(expression_host, expression_, sizeof(GTreeNode) * size_, cudaMemcpyDeviceToHost));

	for (int i = 0; i < size_; i++)
		output << printNode(expression_host[i], i);

	std::free(expression_host);

	output << "End of expression" << std::endl;

	std::string retval(output.str());

	return retval;
}


std::string GExpression::printNode(GTreeNode node, int index) const
{
	std::ostringstream output;

	output << "[" << index << "]: ";
	switch (node.type) {
	case EXPRESSION_TYPE_OPERATOR_PLUS: {
		output << "Operator PLUS";
		break;
	}
	case EXPRESSION_TYPE_OPERATOR_MINUS: {
		output << "Operator MINUS";
		break;
	}
	case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
		output << "Operator MULTIPLY";
		break;
	}
	case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
		output << "Operator DIVIDE";
		break;
	}
	case EXPRESSION_TYPE_OPERATOR_NOT: {
		output << "Operator NOT";
		break;
	}
	case EXPRESSION_TYPE_OPERATOR_CONCAT:
	case EXPRESSION_TYPE_OPERATOR_MOD:
	case EXPRESSION_TYPE_OPERATOR_CAST:
	case EXPRESSION_TYPE_OPERATOR_IS_NULL:
	case EXPRESSION_TYPE_COMPARE_LIKE:
	case EXPRESSION_TYPE_COMPARE_IN: {
		output << "Operator unsupported";
		break;
	}
	case EXPRESSION_TYPE_COMPARE_EQUAL: {
		output << "Compare EQUAL";
		break;
	}
	case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
		output << "Compare NOTEQUAL";
		break;
	}
	case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
		output << "Compare LESSTHAN";
		break;
	}
	case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
		output << "Compare GREATERTHAN";
		break;
	}
	case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
		output << "Compare LESSTHANOREQUALTO";
		break;
	}
	case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
		output << "Compare GREATERTHANOREQUALTO";
		break;
	}
	case EXPRESSION_TYPE_CONJUNCTION_AND: {
		output << "Conjunction AND";
		break;
	}
	case EXPRESSION_TYPE_CONJUNCTION_OR: {
		output << "Conjunction OR";
		break;
	}
	case EXPRESSION_TYPE_VALUE_CONSTANT: {
		output << "Value CONSTANT";
		break;
	}
	case EXPRESSION_TYPE_VALUE_PARAMETER: {
		output << "Value PARAMETER";
		break;
	}
	case EXPRESSION_TYPE_VALUE_TUPLE: {
		output << "Value TUPLE : ";
		output << "Column: " << node.column_idx << ":";
		output << "Table: " << node.tuple_idx;
		break;
	}
	case EXPRESSION_TYPE_VALUE_TUPLE_ADDRESS: {
		output << "Value TUPLE ADDRESS";
		break;
	}
	case EXPRESSION_TYPE_VALUE_NULL: {
		output << "Value NULL";
		break;
	}
	case EXPRESSION_TYPE_VALUE_VECTOR: {
		output << "Value VECTOR";
		break;
	}
	case EXPRESSION_TYPE_AGGREGATE_COUNT: {
		output << "Aggregate COUNT";
		break;
	}
	case EXPRESSION_TYPE_AGGREGATE_COUNT_STAR: {
		output << "Aggregate COUNT STAR";
		break;
	}
	case EXPRESSION_TYPE_AGGREGATE_SUM: {
		output << "Aggregate SUM";
		break;
	}
	case EXPRESSION_TYPE_AGGREGATE_MIN: {
		output << "Aggregate MIN";
		break;
	}
	case EXPRESSION_TYPE_AGGREGATE_MAX: {
		output << "Aggregate MAX";
		break;
	}
	case EXPRESSION_TYPE_AGGREGATE_AVG: {
		output << "Aggregate AVG";
		break;
	}
	case EXPRESSION_TYPE_FUNCTION: {
		output << "FUNCTION";
		break;
	}
	case EXPRESSION_TYPE_HASH_RANGE: {
		output << "HASH RANGE";
		break;
	}
	case EXPRESSION_TYPE_OPERATOR_CASE_WHEN: {
		output << "Operator CASE WHEN";
		break;
	}
	case EXPRESSION_TYPE_OPERATOR_ALTERNATIVE: {
		output << "Operator ALTERNATIVE";
		break;
	}
	case EXPRESSION_TYPE_INVALID:
	default: {
		output << "Invalid node";
		break;
	}
	}
	output << std::endl;

	std::string retval(output.str());

	return retval;
}

GExpressionVector::GExpressionVector(std::vector<ExpressionNode*> expression_list)
{
	if (expression_list.size() > 0) {
		exp_num_ = expression_list.size();

		int *exp_size_host = (int*)malloc(sizeof(int) * (exp_num_ + 1));
		int old_size = 0;

		for (int i = 0; i < exp_num_; i++) {
			exp_size_host[i] = old_size;
			old_size += GExpression::getExpressionLength(expression_list[i]);
		}

		exp_size_host[exp_num_] = old_size;

		checkCudaErrors(cudaMalloc(&exp_size_, sizeof(int) * (exp_num_ + 1)));
		checkCudaErrors(cudaMemcpy(exp_size_, exp_size_host, sizeof(int) * (exp_num_ + 1), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc(&expression_, sizeof(GTreeNode) * old_size));

		for (int i = 0; i < exp_num_; i++) {
			GExpression exp(expression_ + exp_size_host[i], exp_size_host[i + 1] - exp_size_host[i]);

			exp.createExpression(expression_list[i]);
		}

		std::free(exp_size_host);
	} else {
		expression_ = NULL;
		exp_size_ = NULL;
		exp_num_ = 0;
	}
}

void GExpressionVector::free()
{
	if (expression_ != NULL) {
		checkCudaErrors(cudaFree(expression_));
		expression_ = NULL;
	}

	if (exp_size_ != NULL) {
		checkCudaErrors(cudaFree(exp_size_));
		exp_size_ = NULL;
	}

	exp_num_ = 0;
}

std::string GExpressionVector::debug() const
{
	std::ostringstream output;

	output << "Expression list:" << std::endl;
	for (int i = 0; i < exp_num_; i++) {
		GExpression tmp = this->at(i);
		output << "Expression[" << i << "]:" << std::endl << tmp.debug();
	}

	output << "End of the expression list" << std::endl;

	std::string retval(output.str());

	return retval;
}

}
