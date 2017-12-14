/* This file is part of VoltDB.
 * Copyright (C) 2008-2015 VoltDB Inc.
 *
 * This file contains original code and/or modifications of original code.
 * Any modifications made by VoltDB Inc. are licensed under the following
 * terms and conditions:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with VoltDB.  If not, see <http://www.gnu.org/licenses/>.
 */
/* Copyright (C) 2008 by H-Store Project
 * Brown University
 * Massachusetts Institute of Technology
 * Yale University
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#include <vector>
#include <string>
#include <stack>
#include "nestloopexecutor.h"
#include "common/debuglog.h"
#include "common/common.h"
#include "common/tabletuple.h"
#include "common/FatalException.hpp"
#include "common/types.h"
#include "executors/aggregateexecutor.h"
#include "execution/ProgressMonitorProxy.h"
#include "expressions/abstractexpression.h"
#include "expressions/tuplevalueexpression.h"
#include "expressions/comparisonexpression.h"
#include "storage/table.h"
#include "storage/temptable.h"
#include "storage/tableiterator.h"
#include "plannodes/nestloopnode.h"
#include "plannodes/limitnode.h"
#include "plannodes/aggregatenode.h"


#ifdef VOLT_DEBUG_ENABLED
#include <ctime>
#include <sys/times.h>
#include <unistd.h>
#endif

#include "GPUetc/executors/gpunij.h"


using namespace std;
using namespace voltdb;

bool NestLoopExecutor::p_init(AbstractPlanNode* abstract_node,
                              TempTableLimits* limits)
{
    VOLT_TRACE("init NestLoop Executor");

    NestLoopPlanNode* node = dynamic_cast<NestLoopPlanNode*>(abstract_node);
    assert(node);

    // Create output table based on output schema from the plan
    setTempOutputTable(limits);

    assert(m_tmpOutputTable);

    // NULL tuple for outer join
    if (node->getJoinType() == JOIN_TYPE_LEFT) {
        Table* inner_table = node->getInputTable(1);
        assert(inner_table);
        m_null_tuple.init(inner_table->schema());
    }

    // Inline aggregation can be serial, partial or hash
    m_aggExec = voltdb::getInlineAggregateExecutor(m_abstractNode);

    return true;
}

bool NestLoopExecutor::p_execute(const NValueArray &params) {
	std::cout << "Non-indexed Nested Loop Executor" << std::endl;
    VOLT_DEBUG("executing NestLoop...");

    NestLoopPlanNode* node = dynamic_cast<NestLoopPlanNode*>(m_abstractNode);
    assert(node);
    assert(node->getInputTableCount() == 2);

    // output table must be a temp table
    assert(m_tmpOutputTable);
    GTable output = m_tmpOutputTable->getGTable();

    Table* outer_table = node->getInputTable();
    assert(outer_table);
    GTable outer = outer_table->getGTable();

    Table* inner_table = node->getInputTable(1);
    assert(inner_table);
    GTable inner = inner_table->getGTable();
    ExpressionNode *pre_join_exp, *join_exp, *where_exp;

    pre_join_exp = join_exp = where_exp = NULL;

    if (node->getPreJoinPredicate() != NULL)
    	pre_join_exp = node->getPreJoinPredicate()->convertPredicate();

    if (node->getJoinPredicate() != NULL)
    	join_exp = node->getJoinPredicate()->convertPredicate();

    if (node->getWherePredicate() != NULL)
    	where_exp = node->getWherePredicate()->convertPredicate();

    LimitPlanNode* limit_node = dynamic_cast<LimitPlanNode*>(node->getInlinePlanNode(PLAN_NODE_TYPE_LIMIT));
    int limit = -1;
    //int tuple_ctr = 0;
    int offset = -1;
    if (limit_node) {
        limit_node->getLimitAndOffsetByReference(params, limit, offset);
    }

    ProgressMonitorProxy pmp(m_engine, this, inner_table);

    TableTuple join_tuple;
    if (m_aggExec != NULL) {
        VOLT_TRACE("Init inline aggregate...");
        const TupleSchema * aggInputSchema = node->getTupleSchemaPreAgg();
        join_tuple = m_aggExec->p_execute_init(params, &pmp, aggInputSchema, m_tmpOutputTable);
    } else {
        join_tuple = m_tmpOutputTable->tempTuple();
    }

    bool ret;
//    bool earlyReturned = false;
    GPUNIJ gn(outer, inner, &output, pre_join_exp, join_exp, where_exp);

    ret = gn.execute();

    if (!ret) {
    	printf("Error: join failed\n");
    } else {
    	std::cout << "Size of result: " << output.getTupleCount() << std::endl;
    	m_tmpOutputTable->setGTable(output);

//    	result_size = gn.getResultSize();
//    	join_result = (RESULT *)malloc(sizeof(RESULT) * result_size);
//    	gn.getResult(join_result);
//
//    	printf("Result size = %d\n", result_size);
//		for (int i = 0; i < result_size && (limit == -1 || tuple_ctr < limit); i++, tuple_ctr++) {
////			int l = join_result[i].lkey;
////			int r = join_result[i].rkey;
////
//////			join_tuple.setNValues(0, tmp_outer_tuple[l], 0, outer_cols);
////			join_tuple.setNValues(outer_cols, tmp_inner_tuple[r], 0, inner_cols);
//
//			if (m_aggExec != NULL) {
//				if (m_aggExec->p_execute_tuple(join_tuple)){
//					earlyReturned = true;
//					break;
//				}
//			} else {
//				m_tmpOutputTable->insertTempTuple(join_tuple);
//				pmp.countdownProgress();
//			}
//
//			if (earlyReturned) {
//				break;
//			}
//		}
//    }
//
//    if (m_aggExec != NULL) {
//        m_aggExec->p_execute_finish();
    }

    cleanupInputTempTable(inner_table);
    cleanupInputTempTable(outer_table);

    return (true);
}

