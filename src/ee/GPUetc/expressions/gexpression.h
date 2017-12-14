#ifndef GEXPRESSION_H_
#define GEXPRESSION_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/common/common.h"
#include "GPUetc/storage/gnvalue.h"
#include "GPUetc/storage/gtuple.h"
#include <vector>

namespace voltdb {

class GExpression {
public:
	typedef struct _TreeNode {
		ExpressionType type;	//type of
		int column_idx;			//Index of column in tuple, -1 if not tuple value
		int tuple_idx;			//0: left, outer, 1: right, inner
		GNValue value;			// Value of const, = NULL if not const
	} GTreeNode;

	typedef struct _ExpressionNode ExpressionNode;

	struct _ExpressionNode {
		ExpressionNode *left, *right;
		GTreeNode node;
	};


	CUDAH GExpression() {
		expression_ = NULL;
		size_ = 0;
		height_ = 0;
	}

	/* Create a new expression, allocate the GPU memory for
	 * the expression and convert the input pointer-based
	 * tree expression to the desired expression form.
	 */
	GExpression(ExpressionNode *expression);

	/* Create a new expression from an existing GPU buffer. */
	CUDAH GExpression(GTreeNode *expression, int size) {
		expression_ = expression;
		size_ = size;
		height_ = 0;
	}

	/* Create an expression from an input pointer-based tree expression */
	bool createExpression(ExpressionNode *expression);

	void free();

	CUDAH int size()
	{
		return size_;
	}

	CUDAH int height()
	{
		return height_;
	}

	CUDAD GNValue evaluate(int64_t *outer_tuple, int64_t *inner_tuple, GColumnInfo *outer_schema, GColumnInfo *inner_schema, GNValue *stack, int offset)
	{
		int top = 0;

		for (int i = 0; i < size_; i++) {
			GTreeNode tmp = expression_[i];

			switch (tmp.type) {
				case EXPRESSION_TYPE_VALUE_TUPLE: {
					if (tmp.tuple_idx == 0) {
						stack[top] = GNValue(outer_schema[tmp.column_idx].data_type, outer_tuple[tmp.column_idx]);
						top += offset;
					} else if (tmp.tuple_idx == 1) {
						stack[top] = GNValue(inner_schema[tmp.column_idx].data_type, inner_tuple[tmp.column_idx]);
						top += offset;
					}
					break;
				}
				case EXPRESSION_TYPE_VALUE_CONSTANT:
				case EXPRESSION_TYPE_VALUE_PARAMETER: {
					stack[top] = tmp.value;
					top += offset;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_AND: {
					stack[top - 2 * offset] = stack[top - 2 * offset] && stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_OR: {
					stack[top - 2 * offset] = stack[top - 2 * offset] || stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] == stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] != stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
					stack[top - 2 * offset] = stack[top - 2 * offset] < stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
					stack[top - 2 * offset] = stack[top - 2 * offset] <= stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
					stack[top - 2 * offset] = stack[top - 2 * offset] > stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
					stack[top - 2 * offset] = stack[top - 2 * offset] >= stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_PLUS: {
					stack[top - 2 * offset] = stack[top - 2 * offset] + stack[top - offset];
					top -= offset;

					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MINUS: {
					stack[top - 2 * offset] = stack[top - 2 * offset] - stack[top - offset];
					top -= offset;

					break;
				}
				case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
					stack[top - 2 * offset] = stack[top - 2 * offset] / stack[top - offset];
					top -= offset;

					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
					stack[top - 2 * offset] = stack[top - 2 * offset] * stack[top - offset];
					top -= offset;

					break;
				}
				default: {
					return GNValue::getFalse();
				}
			}
		}

		return stack[0];
	}

	CUDAD GNValue evaluate(GTuple outer_tuple, GTuple inner_tuple, GNValue *stack, int offset)
	{
		int top = 0;

		for (int i = 0; i < size_; i++) {
			GTreeNode tmp = expression_[i];

			switch (tmp.type) {
				case EXPRESSION_TYPE_VALUE_TUPLE: {
					if (tmp.tuple_idx == 0) {
						stack[top] = outer_tuple[tmp.column_idx];
						top += offset;
					} else if (tmp.tuple_idx == 1) {
						stack[top] = inner_tuple[tmp.column_idx];
						top += offset;
					}
					break;
				}
				case EXPRESSION_TYPE_VALUE_CONSTANT:
				case EXPRESSION_TYPE_VALUE_PARAMETER: {
					stack[top] = tmp.value;
					top += offset;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_AND: {
					stack[top - 2 * offset] = stack[top - 2 * offset] && stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_OR: {
					stack[top - 2 * offset] = stack[top - 2 * offset] || stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] == stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] != stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
					stack[top - 2 * offset] = stack[top - 2 * offset] < stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
					stack[top - 2 * offset] = stack[top - 2 * offset] <= stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
					stack[top - 2 * offset] = stack[top - 2 * offset] > stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
					stack[top - 2 * offset] = stack[top - 2 * offset] >= stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_PLUS: {
					stack[top - 2 * offset] = stack[top - 2 * offset] + stack[top - offset];
					top -= offset;

					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MINUS: {
					stack[top - 2 * offset] = stack[top - 2 * offset] - stack[top - offset];
					top -= offset;

					break;
				}
				case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
					stack[top - 2 * offset] = stack[top - 2 * offset] / stack[top - offset];
					top -= offset;

					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
					stack[top - 2 * offset] = stack[top - 2 * offset] * stack[top - offset];
					top -= offset;

					break;
				}
				default: {
					return GNValue::getFalse();
				}
			}
		}

		return stack[0];
	}


	CUDAD GNValue evaluate(GTuple outer_tuple, GTuple inner_tuple, GStack stack)
	{
		for (int i = 0; i < size_; i++) {
			GTreeNode tmp = expression_[i];

			switch (tmp.type) {
				case EXPRESSION_TYPE_VALUE_TUPLE: {
					if (tmp.tuple_idx == 0) {
						stack.push(outer_tuple[tmp.column_idx]);
					} else if (tmp.tuple_idx == 1) {
						stack.push(inner_tuple[tmp.column_idx]);
					}
					break;
				}
				case EXPRESSION_TYPE_VALUE_CONSTANT:
				case EXPRESSION_TYPE_VALUE_PARAMETER: {
					stack.push(tmp.value);
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_AND: {
					stack.push(stack.pop() && stack.pop());
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_OR: {
					stack.push(stack.pop() || stack.pop());
					break;
				}
				case EXPRESSION_TYPE_COMPARE_EQUAL: {
					stack.push(stack.pop() == stack.pop());
					break;
				}
				case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
					stack.push(stack.pop() != stack.pop());
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
					stack.push(stack.pop() < stack.pop());
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
					stack.push(stack.pop() <= stack.pop());
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
					stack.push(stack.pop() > stack.pop());
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
					stack.push(stack.pop() >= stack.pop());
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_PLUS: {
					stack.push(stack.pop() + stack.pop());
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MINUS: {
					stack.push(stack.pop() - stack.pop());
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
					stack.push(stack.pop() / stack.pop());
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
					stack.push(stack.pop() * stack.pop());
					break;
				}
				default: {
					return GNValue::getFalse();
				}
			}
		}

		return stack.pop();
	}


	static int getExpressionLength(ExpressionNode *expression);

	std::string debug() const;
private:


	bool buildPostExpression(GTreeNode *output_expression, ExpressionNode *expression, int *index);

	int computeHeight(ExpressionNode *expression);

	std::string printNode(GTreeNode node, int index) const;

	GTreeNode *expression_;
	int size_;
	int height_;
};

class GExpressionVector {
	using GExpression::GTreeNode;
	using GExpression::ExpressionNode;
public:
	CUDAH GExpressionVector();
	GExpressionVector(std::vector<ExpressionNode*> expression_list);
	CUDAH GExpressionVector(GTreeNode *expression_list, int *exp_size, int exp_num);

	CUDAH int size() const;
	CUDAH GExpression at(int exp_idx) const;
	CUDAH GExpression operator[](int exp_idx) const;

	void free();

	std::string debug() const;
private:
	GTreeNode *expression_;
	int *exp_size_;
	int exp_num_;
};

CUDAH GExpressionVector::GExpressionVector()
{
	expression_ = NULL;
	exp_size_ = NULL;
	exp_num_ = 0;
}

CUDAH GExpressionVector::GExpressionVector(GTreeNode *expression_list, int *exp_size, int exp_num)
{
	expression_ = expression_list;
	exp_size_ = exp_size;
	exp_num_ = exp_num;
}

CUDAH int GExpressionVector::size() const
{
	return exp_num_;
}

CUDAH GExpression GExpressionVector::at(int exp_idx) const
{
	if (exp_idx >= exp_num_)
		return GExpression();

	return GExpression(expression_ + exp_size_[exp_idx], exp_size_[exp_idx + 1] - exp_size_[exp_idx]);
}

CUDAH GExpression GExpressionVector::operator[](int exp_idx) const
{
	if (exp_idx >= exp_num_)
		return GExpression();

	return GExpression(expression_ + exp_size_[exp_idx], exp_size_[exp_idx + 1] - exp_size_[exp_idx]);
}

class GStack {
public:
	CUDAH GStack();

	CUDAH GStack(GNValue *buffer, int stride);

	CUDAH GNValue pop();

	CUDAH void push(GNValue new_val);

	CUDAH void reset();

private:
	GNValue *buffer_;
	int top_;
	int stride_;
};

CUDAH GStack::GStack()
{
	buffer_ = NULL;
	top_ = 0;
	stride_ = 0;
}

CUDAH GStack::GStack(GNValue *buffer, int stride)
{
	buffer_ = buffer;
	top_ = 0;
	stride_ = stride;
}

CUDAH GNValue GStack::pop()
{
	if (top_ > 0)
		top_ -= stride_;

		return buffer_[top_];

	return GNValue::getInvalid();
}

CUDAH void GStack::push(GNValue new_val)
{
	buffer_[top_] = new_val;
	top_ += stride_;
}

CUDAH void GStack::reset()
{
	top_ = 0;
}

}

#endif
