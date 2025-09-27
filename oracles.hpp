#include "fft.h"


#ifndef __CUDA_ARCH__
#define __PLACE__ 
#else
#define __PLACE__ __device__
#endif

__PLACE__ static constexpr uint64_t MDSE[8][8] = {
    {0xa, 0xe, 0x2, 0x6, 0x5, 0x7, 0x1, 0x3},
    {0x8, 0xc, 0x2, 0x2, 0x4, 0x6, 0x1, 0x1},
    {0x2, 0x6, 0xa, 0xe, 0x1, 0x3, 0x5, 0x7},
    {0x2, 0x2, 0x8, 0xc, 0x1, 0x1, 0x4, 0x6},
    {0x5, 0x7, 0x1, 0x3, 0xa, 0xe, 0x2, 0x6},
    {0x4, 0x6, 0x1, 0x1, 0x8, 0xc, 0x2, 0x2},
    {0x1, 0x3, 0x5, 0x7, 0x2, 0x6, 0xa, 0xe},
    {0x1, 0x1, 0x4, 0x6, 0x2, 0x2, 0x8, 0xc},
};

static constexpr uint64_t MDSE_inv[8][8] = {
    {0x1555555540000000, 0xbfffffff40000001, 0x7fffffff80000000, 0x6aaaaaaa40000001, 0xf555555460000001, 0x1fffffffe0000000, 0xbfffffff40000001, 0x4aaaaaaa60000000},
    {0xeaaaaaa9c0000001, 0x1555555540000000, 0xaaaaaaaa00000001, 0x6aaaaaaa40000000, 0xaaaaaaaa0000000,  0xf555555460000001, 0x2aaaaaaa80000000, 0xcaaaaaa9e0000001},
    {0x7fffffff80000000, 0x6aaaaaaa40000001, 0x1555555540000000, 0xbfffffff40000001, 0xbfffffff40000001, 0x4aaaaaaa60000000, 0xf555555460000001, 0x1fffffffe0000000},
    {0xaaaaaaaa00000001, 0x6aaaaaaa40000000, 0xeaaaaaa9c0000001, 0x1555555540000000, 0x2aaaaaaa80000000, 0xcaaaaaa9e0000001, 0xaaaaaaaa0000000,  0xf555555460000001},
    {0xf555555460000001, 0x1fffffffe0000000, 0xbfffffff40000001, 0x4aaaaaaa60000000, 0x1555555540000000, 0xbfffffff40000001, 0x7fffffff80000000, 0x6aaaaaaa40000001},
    {0xaaaaaaaa0000000,  0xf555555460000001, 0x2aaaaaaa80000000, 0xcaaaaaa9e0000001, 0xeaaaaaa9c0000001, 0x1555555540000000, 0xaaaaaaaa00000001, 0x6aaaaaaa40000000},
    {0xbfffffff40000001, 0x4aaaaaaa60000000, 0xf555555460000001, 0x1fffffffe0000000, 0x7fffffff80000000, 0x6aaaaaaa40000001, 0x1555555540000000, 0xbfffffff40000001},
    {0x2aaaaaaa80000000, 0xcaaaaaa9e0000001, 0xaaaaaaaa0000000,  0xf555555460000001, 0xaaaaaaaa00000001, 0x6aaaaaaa40000000, 0xeaaaaaa9c0000001, 0x1555555540000000},
};

constexpr uint64_t MDSE_inv77_inv = 0xfffffffefffffff5;
constexpr uint64_t MDSE_inv72_inv = 0xfffffffeffffffe9;
constexpr uint64_t MDSE_inv70_neg = 0xd555555480000001;
constexpr uint64_t MDSE_inv71_neg = 0x3555555520000000;
constexpr uint64_t root_exp = 0x92492491b6db6db7;

inline void linearE(uint64_t state[8]) {
    uint64_t tmp[8] = {};
    for (long i = 0; i < 8; i++) {
        for (long j = 0; j < 8; j++) {
            tmp[i] = fft64::add(tmp[i], fft64::mul(MDSE[i][j], state[j]));
        }
    }
    for (long i = 0; i < 8; i++) state[i] = tmp[i];
}

inline void sbox(uint64_t &x) {
    uint64_t x2 = fft64::mul(x, x);
    uint64_t x4 = fft64::mul(x2, x2);
    x = fft64::mul(x, fft64::mul(x2, x4));
}

static constexpr inline uint64_t compute_LX1(uint64_t seed) {
    return seed;
}

static constexpr inline uint64_t compute_LX2(uint64_t seed) {
    uint64_t LX2 = fft64::add(MDSE_inv70_neg, fft64::mul(MDSE_inv71_neg, fft64::pow(seed, root_exp)));
    LX2 = fft64::pow(fft64::mul(LX2, MDSE_inv72_inv), 7);
    return LX2;
}

template <class Oracle>
static constexpr inline uint64_t compute_CX7(uint64_t seed) {
    uint64_t CX7 = 0;
    for (long i = 0; i < 8; i++) {
        CX7 = fft64::add(CX7, fft64::mul(MDSE_inv[7][i], Oracle::get_RC(i)));
    }
    CX7 = fft64::pow(fft64::mul(MDSE_inv77_inv, CX7), 7);
    return CX7;
}

template <class Oracle>
static inline void linearI(uint64_t state[8]) {
    uint64_t sum = 0;
    for (long i = 0; i < 8; i++) {
        sum = fft64::add(sum, state[i]);
    }
    for (long i = 0; i < 8; i++) {
        state[i] = fft64::add(sum, fft64::mul(Oracle::get_MDSI(i), state[i]));
    }
}

template <class Oracle>
static void cpu_eval(uint64_t *st, uint64_t num) {
    for (long l = 0; l < num; l++) {
        uint64_t state[8];
        state[0] = st[l];
        state[1] = fft64::mul(Oracle::LX1, st[l]);
        state[2] = fft64::mul(Oracle::LX2, st[l]);
        state[3] = state[4] = state[5] = state[6] = 0;
        state[7] = Oracle::CX7;

        int RC_counter = 8;

        linearE(state);

        for (uint64_t r = 0; r < Oracle::R_F / 2 - 1; r++) {
            for (uint64_t i = 0; i < 8; i++) {
                state[i] = fft64::add(state[i], Oracle::get_RC(RC_counter++));
            }
            for (uint64_t i = 0; i < 8; i++) {
                sbox(state[i]);
            }
            linearE(state);
        }
        for (uint64_t r = 0; r < Oracle::R_P; r++) {
            state[0] = fft64::add(state[0], Oracle::get_RC(RC_counter++));
            sbox(state[0]);
            linearI<Oracle>(state);
        }

        for (uint64_t r = 0; r < Oracle::R_F / 2; r++) {
            for (uint64_t i = 0; i < 8; i++) {
                state[i] = fft64::add(state[i], Oracle::get_RC(RC_counter++));
            }
            for (uint64_t i = 0; i < 8; i++) {
                sbox(state[i]);
            }
            linearE(state);
        }
        st[l] = state[7];
    }
}

template <class Oracle>
static void compute_input_tmpl(uint64_t st[8], uint64_t x) {
    st[0] = x;
    st[1] = fft64::mul(Oracle::LX1, x);
    st[2] = fft64::mul(Oracle::LX2, x);
    st[3] = st[4] = st[5] = st[6] = 0;
    st[7] = Oracle::CX7;
    for (long i = 0; i < 8; i++) st[i] = fft64::pow(st[i], root_exp);
    for (long i = 0; i < 8; i++) st[i] = fft64::sub(st[i], Oracle::get_RC(i));
    uint64_t tmp[8] = {};
    for (long i = 0; i < 8; i++) {
        for (long j = 0; j < 8; j++) {
            tmp[i] = fft64::add(tmp[i], fft64::mul(MDSE_inv[i][j], st[j]));
        }
    }
    for (long i = 0; i < 8; i++) st[i] = tmp[i];
}


template <uint64_t seed>
struct Poseidon2_perm64_24b {
    static constexpr uint64_t R_F = 6;
    static constexpr uint64_t R_P = 7;
    static constexpr uint64_t degree = 13841287201;

    static constexpr uint64_t LX1 = compute_LX1(seed);
    static constexpr uint64_t LX2 = compute_LX2(seed);
    static constexpr uint64_t CX7 = compute_CX7<Poseidon2_perm64_24b<seed>>(seed);
    static __PLACE__ constexpr uint64_t get_RC(int i) {
        constexpr uint64_t _RC[8 * R_F + R_P] = {
            0xd879e902dbfd1007, 0x38563021491e933b, 0x0d3cfee2257d9edf, 0x9b52c442009d453e,
            0xd60fe06183a67aef, 0x30379aad4961c5a7, 0xfe764e101738387c, 0x5298238bf7738cac,
            0x8a9cd0f6509d98ad, 0x4c8307b44ad52575, 0x4b1d15a59d8754b4, 0xc88b081034916041,
            0xf1edde2dba74e98c, 0x5dbbce95ab0c6efa, 0xa2e68bb8a310f642, 0xb645ca3cc8ec1833,
            0xb604d6a0824659fb, 0xd6e81dd0eca2fbe5, 0x8b1b1956b7ffdcae, 0x3eceaf5f23c33154,
            0x496e95b1c1b0bd3c, 0x9f605ab0341d0919, 0xe2c2fabb03bc8e9e, 0xcc6f70e0dcb05960,
            0xbe923850bae01bed, 0xeb5006a240b162aa, 0x83fd473afde38fa0, 0x69b50f865794b274,
            0x0938a99246283fef, 0xe30576a05ae0526a, 0xdaecc7c6eb5b645b, 0xd715dad1686d0ae9,
            0x6680c1ae907cc359, 0x63f59bae7f0785ea, 0x86c92332480bcdcb, 0x2f5e5004bf885c61,
            0xcf1c149727da6c40, 0x7ca10fc47c8bd68f, 0x70b7646a7f847a5a, 0xe063740bd70b51b8,
            0xf8aa941c0f036124, 0x341423e70fd417c1, 0xff2434d11cc85db7, 0x998995051ebf5301,
            0xaaef61aa0b31216d, 0x8cf3568ee4f53bd0, 0x8a9269e185847980, 0xe4324102949599f3,
            0xfc380ee08ef3ae46, 0xce4b3355356f1c3a, 0x61612f53bed21968, 0xc4684c31a3f78766,
            0x528f440d555bf255, 0x72aa30239185421d, 0x080456bba80f07ff
        };
        return _RC[i];
    }
    static __PLACE__ constexpr uint64_t get_MDSI(int i) {
        constexpr uint64_t _MDSI[8] = {
            0xa15bcb1cd9bf1355, 0xb6ad2855d39a6c78, 0x9f9f8d65047d55ea, 0x137f0026ccd98fe0,
            0x4636422313ee0e94, 0x06a1d036d6982e85, 0x59f301fd715e8087, 0x793728e092b89f5d
        };
        return _MDSI[i];
    }

    static void eval(uint64_t *st, uint64_t num) {
        cpu_eval<Poseidon2_perm64_24b<seed>>(st, num);
    }

    static void compute_input(uint64_t st[8], uint64_t x) {
        compute_input_tmpl<Poseidon2_perm64_24b<seed>>(st, x);
    }
};

template <uint64_t seed>
struct Poseidon2_perm64_28b {
    static constexpr uint64_t R_F = 6;
    static constexpr uint64_t R_P = 8;
    static constexpr uint64_t degree = 96889010407;

    static constexpr uint64_t LX1 = compute_LX1(seed);
    static constexpr uint64_t LX2 = compute_LX2(seed);
    static constexpr uint64_t CX7 = compute_CX7<Poseidon2_perm64_28b<seed>>(seed);
    static __PLACE__ constexpr uint64_t get_RC(int i) {
        constexpr uint64_t _RC[8 * R_F + R_P] = {
            0x99999819a2514e36, 0x7aa88942f9c47b66, 0x295afd4a2467921e, 0xd47c10eceb442d16,
            0x24d1cf1ec778c2cb, 0xfeda575280285a81, 0xdb9b94c50651f3a8, 0x5728b22ccb83bbfe,
            0x6836cbd9054b332d, 0x24761db787b7aa8b, 0x9a44e7dd2b69ff70, 0xa90c270d632c03a7,
            0xfbb7d6af63407a0d, 0x5a155565997c8f7b, 0x01859c69f8674a6b, 0x29df785b5e8adb34,
            0xa2b429ed8992ba2d, 0x4ad195d35c0a545a, 0x35406c7ffc0e4f1e, 0xa5cf92c4af9ba7f8,
            0xb2479b3ff9837c0b, 0x9d1fbf0aef165021, 0x65842c1489937ce6, 0x29a2f8f24dc86b3c,
            0x0e0e9cc8c1525758, 0x2b666e61540aac82, 0x1d17a125b02ae12a, 0x80c7609ce49e2e7d,
            0x4f524c566dcf6e88, 0xabeed76a57c91d16, 0x916ffe38519e0b2f, 0xb928ddda318db7b0,
            0xb018c3a6d49a6674, 0x375c4309b2434c08, 0xaba76ecd28fa1ec6, 0x94e2f99940b5de74,
            0x98e0c366e59840ac, 0x660531c07d0c6ae3, 0x6b09b266a08218b9, 0xf24616b8d096cfb2,
            0xb1287857678bc049, 0x27e3cea3e412406f, 0x4cd194238a06e040, 0xf99141f936c44276,
            0xe0952c29de42620a, 0xa408c665f6e0b1c9, 0x8c197545ac609081, 0xbeb6f3bce6c13c7d,
            0x73eeac7a19169382, 0x40d85de29ef7aa53, 0xf31da154f01205ee, 0x78253dd4b562834e,
            0x4a7925b110b720c0, 0xca5b7c171e385f0d, 0xc86964b3f59478ff, 0x72f7edc2d0d0dfe6
        };
        return _RC[i];
    }
    static __PLACE__ constexpr uint64_t get_MDSI(int i) {
        constexpr uint64_t _MDSI[8] = {
            0x36100caf9c567a9c, 0x3fc38f32e890a6c2, 0xe3a79a6cc844d1a6, 0x042a98959411d964,
            0xf84e84623ebc3080, 0xaa2d0367d501732e, 0x73fffe20c5596086, 0x66a4d41935cabb99
        };
        return _MDSI[i];
    }

    static void eval(uint64_t *st, uint64_t num) {        
        cpu_eval<Poseidon2_perm64_28b<seed>>(st, num);
    }

    static void compute_input(uint64_t st[8], uint64_t x) {
        compute_input_tmpl<Poseidon2_perm64_28b<seed>>(st, x);
    }
};

template <uint64_t seed>
struct Poseidon2_perm64_32b {
    static constexpr uint64_t R_F = 6;
    static constexpr uint64_t R_P = 10;
    static constexpr uint64_t degree = 4747561509943;

    static constexpr uint64_t LX1 = compute_LX1(seed);
    static constexpr uint64_t LX2 = compute_LX2(seed);
    static constexpr uint64_t CX7 = compute_CX7<Poseidon2_perm64_32b<seed>>(seed);
    static __PLACE__ constexpr uint64_t get_RC(int i) {
        constexpr uint64_t _RC[8 * R_F + R_P] = {
            0x83bdfc2c9301f4ce, 0x94323877b10e3986, 0x8de951a0bcb6628a, 0xb53566cde2858ccd,
            0xaa5cb9ddce7feb54, 0x867270a224f1d107, 0x45e45693dcc77bb2, 0x5f0b6ba359022d8b,
            0xe26beb57dc573dcd, 0x6c08cd8a875bf42d, 0xa4f295ac660bc2a1, 0x378d5541456ac1bd,
            0x8ecb8cff84497c7c, 0x0b3a9a39baca4802, 0xdf9f0f6678e1d322, 0x1d6612ffa5728080,
            0x2591ddcf26942c26, 0x94cf6bea0e34effb, 0x3da95e1953dc30e4, 0xb40a7bafd65ca008,
            0x7fb5b22a62b61384, 0xc9dc65ef74640d46, 0xa7ce11b931f4a9a5, 0x4d950e1b10f1a145,
            0x40eb5eaa25d5dad5, 0x2c12a1dd266fb56c, 0x8723da834097472f, 0x1848666d28022486,
            0xed9749658e2b1b55, 0xe5f6ae4b18b5e4d9, 0x8e78c84cad0da91b, 0xb6366c405c32aa8a,
            0x5699e2d6ba76b48b, 0xff614794921144da, 0x40056c8e67e3fa0d, 0x857803df432a36d1,
            0xd2637f0fc39ab967, 0x86dc484c3c7c56d0, 0xe81d338fd2873d9e, 0xff0b54b1ad6a2cde,
            0xde69a799e08d91ff, 0xdc0b257e8f14169b, 0xab6a4053fbd7a036, 0x903e9b402e34e470,
            0x939c71a75a0d9022, 0x65622505bfce55f6, 0xc8d979fce70e25a1, 0x19e942ab95067fe7,
            0xcc003f98ef209763, 0x0d766689c4de290d, 0x268dceb8f07d4e3c, 0xb40be98132fbd81f,
            0x2addd9e5cdea5ac6, 0x7f8b890bf44e0cf1, 0x824b26fc83d33f01, 0x344f23d8bdd13f64,
            0x68661c6485587d2d, 0xc729b7d2abb88ce6
        };
        return _RC[i];
    }
    static __PLACE__ constexpr uint64_t get_MDSI(int i) {
        constexpr uint64_t _MDSI[8] = {
            0x1dbbd96b2b5205d5, 0x5a3bf5b74fb223ea, 0xd14790a34c3ed8d7, 0x83257f0ca3022b06,
            0x5ca824862789356f, 0x4a7eae9854657e88, 0xf31f7a40f909cdf5, 0x55eda1bc07b44568
        };
        return _MDSI[i];
    }

    static void eval(uint64_t *st, uint64_t num) {        
        cpu_eval<Poseidon2_perm64_32b<seed>>(st, num);
    }

    static void compute_input(uint64_t st[8], uint64_t x) {
        compute_input_tmpl<Poseidon2_perm64_32b<seed>>(st, x);
    }
};