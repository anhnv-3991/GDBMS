#ifndef GPUIJ_H
#define GPUIJ_H

#include <cuda.h>
#include "common/types.h"
#include "GPUetc/storage/gtable.h"
#include "GPUetc/expressions/gexpression.h"

namespace voltdb {


class GPUIJ {
	using GExpression::ExpressionNode;

public:
	GPUIJ();

	GPUIJ(GTable outer_table,
			GTable inner_table,
			std::vector<ExpressionNode*> search_idx,
			ExpressionNode *end_expression,
			ExpressionNode *post_expression,
			ExpressionNode *initial_expression,
			ExpressionNode *skipNullExpr,
			ExpressionNode *prejoin_expression,
			ExpressionNode *where_expression,
			IndexLookupType lookup_type);

	~GPUIJ();

	bool execute();

	void getResult(RESULT *output) const;

	int getResultSize() const;

	void debug();

private:
	GTable outer_table_, inner_table_;
	GTable search_table_;
	RESULT *join_result_;
	int result_size_;
	IndexLookupType lookup_type_;

	GExpressionVector search_exp_;
	GExpression end_expression_;
	GExpression post_expression_;
	GExpression initial_expression_;
	GExpression skipNullExpr_;
	GExpression prejoin_expression_;
	GExpression where_expression_;

	//For profiling
	std::vector<unsigned long> allocation_, prejoin_, index_, expression_, ipsum_, epsum_, wtime_, joins_only_, rebalance_;
	struct timeval all_start_, all_end_;

	void profiling();

	uint getPartitionSize() const;

	unsigned long timeDiff(struct timeval start, struct timeval end);

	void prejoinFilter(bool *result);
	void prejoinFilter(bool *result, cudaStream_t stream);

	void decompose(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size);
	void decompose(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size, cudaStream_t stream);

	void indexFilter(ulong *index_psum, ResBound *res_bound, bool *prejoin_res_dev);

	void indexFilter(ulong *index_psum, ResBound *res_bound, bool *prejoin_res_dev, cudaStream_t stream);

	/* Expression evaluation without rebalancing */
	void expressionFilter(ulong *index_psum, ulong *exp_psum, RESULT *result, int result_size, ResBound *res_bound, bool *prejoin_res_dev);

	void expressionFilter(ulong *index_psum, ulong *exp_psum, RESULT *result, int result_size, ResBound *res_bound, bool *prejoin_res_dev, cudaStream_t stream);

	/* Expression evaluation with rebalancing */
	void expressionFilter(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size);

	void expressionFilter(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size, cudaStream_t stream);

	void rebalance(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size);
	void rebalance(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);
};
}

#endif
