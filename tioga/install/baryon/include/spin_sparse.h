#ifndef SPIN_SPARSE_H
#define SPIN_SPARSE_H

#include <iostream>

using namespace std;
using cd = complex<double>;

// rank-4 spin in sparse representation
typedef struct{
    int s[Nc];
    cd val;
} Spin4;

#endif
