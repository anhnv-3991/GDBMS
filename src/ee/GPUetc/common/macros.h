#ifndef GPUTUPLE_H
#define GPUTUPLE_H

namespace voltdb{

#define DEFAULT_PART_SIZE_ (1024 * 1024)
//#define DEFAULT_PART_SIZE_ 1024
//#define DEFAULT_PART_SIZE_ (128 * 1024)
#define PART_SIZE_ 1024
//1blockでのスレッド数の定義。
//#define BLOCK_SIZE_X 1024//outer ,left
#define BLOCK_SIZE_X 1024
#define GRID_SIZE_X (1024 * 1024)
#define GRID_SIZE_Y (64 * 1024)

//#define BLOCK_SIZE_Y 2048  //inner ,right
#define BLOCK_SIZE_Y (1024 * 1024)


#define PARTITION 64
#define RADIX 6
#define PART_C_NUM 16
#define SHARED_MAX PARTITION * PART_C_NUM
#define MAX_BLOCK_SIZE (1024 * 1024 * 32)

#define MAX_ROWS_PER_BLOCK (1048576)
#define MAX_COLS_PER_BLOCK (16)
#define ADDITIONAL_COLS (4)

#define RIGHT_PER_TH 256

#define PART_STANDARD 1
#define JOIN_SHARED 256
#define MAX_GNVALUE 10
#define MAX_STACK_SIZE 32
#define MAX_SHARED_MEM 16
#define MAX_BUFFER_SIZE (1024 * 1024)
#define SHARED_MEM 128
#define SHARED_SIZE_ 49152


typedef struct _RESULT {
    int lkey;
    int rkey;
} RESULT;

typedef struct _RESULT_BOUND {
	int outer;
	int left;
	int right;
} ResBound;


}

#endif
