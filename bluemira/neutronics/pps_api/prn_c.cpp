#include "openmc/random_lcg.h"

extern "C" double prn_c(uint64_t* seed){

    return openmc::prn(seed);

}
