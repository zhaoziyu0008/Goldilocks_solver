#include "poly.h"
#include "omp.h"

#include "oracles.hpp"

int main() {
    omp_set_num_threads(96);

    uint64_t x = root<Poseidon2_perm64_24b<0>, fft64>();
    if (x == -1) {
        printf("no roots\n");
    } else {
        uint64_t st[8];
        Poseidon2_perm64_24b<0>::compute_input(st, x);
        for (int i = 0; i < 8; i++) {
            printf("%lx ", st[i]);
        }
        printf("\n");
    }


    return 0;
}