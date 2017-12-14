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

#include "temptable.h"
#include "common/debuglog.h"

#define TABLE_BLOCKSIZE 131072

namespace voltdb {

TempTable::TempTable()
  : Table(TABLE_BLOCKSIZE),
    m_iter(this),
    m_limits(NULL)
{
    // this happens here because m_data might not be initialized above
    m_iter.reset(m_data.begin());
}

TempTable::~TempTable() {}

// ------------------------------------------------------------------
// OPERATIONS
// ------------------------------------------------------------------
void TempTable::deleteAllTuples(bool freeAllocatedStrings) {
    deleteAllTuplesNonVirtual(freeAllocatedStrings);
}

bool TempTable::insertTuple(TableTuple &source) {
    insertTempTuple(source);
    //Added for GPU
//	int64_t *gtuple = (int64_t*)malloc(sizeof(int64_t) * m_columnCount);
//
//	NValue tmp_value;
//	GNValue tmp_gvalue;
//	ValueType tmp_type;
//
//	for (int i = 0; i < m_columnCount; i++) {
//		tmp_value = source.getNValue(i);
//		tmp_type = tmp_value.getValueTypeForGPU();
//
//		tmp_gvalue.setValue(tmp_type, tmp_value.getValueForGPU());
//		*(gtuple + i) = tmp_gvalue.getValue();
//	}
//
//	m_gtable.insertTuple(gtuple);
//
//	free(gtuple);
    return true;
}

bool TempTable::updateTupleWithSpecificIndexes(TableTuple &targetTupleToUpdate,
                                               TableTuple &sourceTupleWithNewValues,
                                               std::vector<TableIndex*> const &indexesToUpdate,
                                               bool)
{
    throwFatalException("TempTable does not support update");
    // Some day maybe, if we find a use case:
    // Copy the source tuple into the target
    // targetTupleToUpdate.copy(sourceTupleWithNewValues);
}

bool TempTable::deleteTuple(TableTuple &, bool)
{
    throwFatalException("TempTable does not support deleting individual tuples");
}

std::string TempTable::tableType() const { return "TempTable"; }

voltdb::TableStats* TempTable::getTableStats() { return NULL; }

}