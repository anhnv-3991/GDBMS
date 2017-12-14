#ifndef NODEDATA_H
#define NODEDATA_H

#include "common/types.h"
//#include "gnvalue.h"

namespace voltdb {

#define CUDAH __forceinline__ __host__ __device__
#define CUDAD __forceinline__ __device__

typedef struct {
	ValueType data_type;
} GColumnInfo;

}

#endif
