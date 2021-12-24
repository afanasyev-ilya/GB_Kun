#ifndef GB_KUN_TYPES_HPP
#define GB_KUN_TYPES_HPP
    #include <cstddef>


    enum Storage {GrB_UNKNOWN,
            GrB_SPARSE,
            GrB_DENSE};

    enum Desc_field {GrB_MASK,
            GrB_OUTP,
            GrB_INP0,
            GrB_INP1,
            GrB_MODE,
            GrB_TA,
            GrB_TB,
            GrB_NT,
            GrB_MXVMODE,
            GrB_TOL,
            GrB_BACKEND,
            GrB_NDESCFIELD};

    enum Desc_value {GrB_SCMP,               // for GrB_MASK
            GrB_REPLACE,            // for GrB_OUTP
            GrB_TRAN,               // for GrB_INP0, GrB_INP1
            GrB_DEFAULT,
            GrB_CUSPARSE,           // for SpMV, SpMM
            GrB_CUSPARSE2,
            GrB_FIXEDROW,
            GrB_FIXEDCOL,
            GrB_MERGEPATH  =    9,
            GrB_PUSHPULL   =   10,  // for GrB_MXVMODE
            GrB_PUSHONLY   =   11,  // for GrB_MXVMODE
            GrB_PULLONLY   =   12,  // for GrB_MXVMODE
            GrB_SEQUENTIAL =   13,  // for GrB_BACKEND
            GrB_CUDA       =   14,  // for GrB_BACKEND
            GrB_8          =    8,  // for GrB_TA, GrB_TB, GrB_NT
            GrB_16         =   16,  // for GrB_TOL
            GrB_32         =   32,
            GrB_64         =   64,
            GrB_128        =  128,
            GrB_256        =  256,
            GrB_512        =  512,
            GrB_1024       = 1024};


    typedef enum {

        GrB_SUCCESS = 0,            // all is well

        //--------------------------------------------------------------------------
        // informational codes, not an error:
        //--------------------------------------------------------------------------

        GrB_NO_VALUE = 1,           // A(i,j) requested but not there

        //--------------------------------------------------------------------------
        // API errors:
        //--------------------------------------------------------------------------

        GrB_UNINITIALIZED_OBJECT = 2,   // object has not been initialized
        GrB_INVALID_OBJECT = 3,         // object is corrupted
        GrB_NULL_POINTER = 4,           // input pointer is NULL
        GrB_INVALID_VALUE = 5,          // generic error code; some value is bad
        GrB_INVALID_INDEX = 6,          // a row or column index is out of bounds
        GrB_DOMAIN_MISMATCH = 7,        // object domains are not compatible
        GrB_DIMENSION_MISMATCH = 8,     // matrix dimensions do not match
        GrB_OUTPUT_NOT_EMPTY = 9,       // output matrix already has values in it

        //--------------------------------------------------------------------------
        // execution errors:
        //--------------------------------------------------------------------------

        GrB_OUT_OF_MEMORY = 10,         // out of memory
        GrB_INSUFFICIENT_SPACE = 11,    // output array not large enough
        GrB_INDEX_OUT_OF_BOUNDS = 12,   // a row or column index is out of bounds
        GrB_PANIC = 13                  // unknown error, or GrB_init not called.
    }
    LA_Info;


//template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
//        struct plus {
//            inline T_out operator()(T_in1 lhs, T_in2 rhs) {
//                return lhs + rhs;
//            }
//        };
namespace lablas{

        template <typename D1, typename D2 = D1>
        struct Identity {
            inline D2 operator()(const D1 input) const { return input; }
        };

        template <typename T_out>
        struct plus {
            inline T_out operator()(const T_out lhs, const T_out rhs) const {
                return lhs + rhs;
            }
        };

        template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
        struct minus {
            inline T_out operator()(const T_in1 lhs, const T_in2 rhs) const {
                return lhs - rhs;
            }
        };

        template <typename T_out>
        struct abs {
            inline T_out operator()(const T_out arg) const {
                return std::abs(arg);
            }
        };

        template <typename T>
        struct div {
            inline T operator()(const T lhs, const T rhs) const {
                return lhs / rhs;
            }
        };

        template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
        struct multiplies {
            inline T_out operator()(const T_in1 lhs, const T_in2 rhs) const {
                return lhs * rhs;
            }
        };

        template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
        struct first {
            inline T_out operator()(const T_in1 lhs, const T_in2 rhs) const {
                return lhs;
            }
        };

        template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
        struct second {
            inline T_out operator()(const T_in1 lhs, const T_in2 rhs) const {
                return lhs;
            }
        };

        template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
        struct minimum {
            inline T_out operator()(const T_in1 lhs, const T_in2 rhs) const {
                if (lhs < rhs){
                    return lhs;
                } else {
                    return rhs;
                }
            }
        };

        template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1>
        struct maximum {
            inline T_out operator()(const T_in1 lhs, const T_in2 rhs) const {
                if (lhs > rhs){
                    return lhs;
                } else {
                    return rhs;
                }
            }
        };

        template <typename T_in1 = bool, typename T_in2 = bool, typename T_out = bool>
        struct logical_and {
            inline T_out operator()(const T_in1 lhs, const T_in2 rhs) const {
                return lhs && rhs;
            }
        };

// Monoid generator macro provided by Scott McMillan.
#define REGISTER_MONOID(M_NAME, BINARYOP, IDENTITY)                          \
template <typename T_out>                                                    \
struct M_NAME                                                                \
{                                                                            \
inline T_out identity() const                                              \
    {                                                                          \
        return static_cast<T_out>(IDENTITY);                                     \
    }                                                                          \
inline T_out operator()(T_out lhs, T_out rhs) const    \
    {                                                                          \
        return BINARYOP<T_out>()(lhs, rhs);                                      \
    }                                                                          \
};

REGISTER_MONOID(PlusMonoid, plus, 0)
REGISTER_MONOID(LogicalOrMonoid, logical_or, false)

REGISTER_MONOID(FirstWinsMonoid, first, 0)
REGISTER_MONOID(SecondWinsMonoid, second, 0)

REGISTER_MONOID(FirstMin, minimum, 0)


// Semiring generator macro provided by Scott McMillan
#define REGISTER_SEMIRING(SR_NAME, ADD_MONOID, MULT_BINARYOP)             \
template <typename T_in1, typename T_in2 = T_in1, typename T_out = T_in1> \
struct SR_NAME                                                            \
{                                                                         \
typedef T_out result_type;                                              \
typedef T_out T_out_type;                                               \
\
inline T_out identity() const                                           \
{ return ADD_MONOID<T_out>().identity(); }                              \
\
inline  T_out add_op(T_out lhs, T_out rhs)           \
{ return ADD_MONOID<T_out>()(lhs, rhs); }                               \
\
inline  T_out mul_op(T_in1 lhs, T_in2 rhs)           \
{ return MULT_BINARYOP<T_in1, T_in2, T_out>()(lhs, rhs); }                \
};


REGISTER_SEMIRING(PlusMultipliesSemiring, PlusMonoid, multiplies)
REGISTER_SEMIRING(LogicalOrAndSemiring, LogicalOrMonoid, logical_and)
REGISTER_SEMIRING(FirstWinsSemiring, FirstWinsMonoid, multiplies)
REGISTER_SEMIRING(FirstMinSemiring, FirstMin, multiplies)
REGISTER_SEMIRING(PlusSecondSemiring, PlusMonoid, second)

template <typename SemiringT>
struct AdditiveMonoidFromSemiring {
public:
    typedef typename SemiringT::T_out_type T_out_type;
    typedef typename SemiringT::T_out_type result_type;

    typedef typename SemiringT::T_out_type first_argument_type;
    typedef typename SemiringT::T_out_type second_argument_type;

    AdditiveMonoidFromSemiring() : sr() {}
    explicit AdditiveMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

    inline T_out_type identity() const {
        return sr.identity();
    }

    template <typename T_in1, typename T_in2>
    inline T_out_type operator()(T_in1 lhs, T_in2 rhs) {
        return sr.add_op(lhs, rhs);
    }

private:
    SemiringT sr;
};

template <typename SemiringT>
struct MultiplicativeMonoidFromSemiring {
public:
    typedef typename SemiringT::T_out_type T_out_type;
    typedef typename SemiringT::T_out_type result_type;

    typedef typename SemiringT::T_out_type first_argument_type;
    typedef typename SemiringT::T_out_type second_argument_type;

    MultiplicativeMonoidFromSemiring() : sr() {}
    explicit MultiplicativeMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

    inline T_out_type identity() const {
        return sr.identity();
    }

    template <typename T_in1, typename T_in2>
    inline T_out_type operator()(T_in1 lhs, T_in2 rhs) {
        return sr.mul_op(lhs, rhs);
    }

private:
    SemiringT sr;
};

template <typename SemiringT>
AdditiveMonoidFromSemiring<SemiringT>
extractAdd(SemiringT const &sr) {
    return AdditiveMonoidFromSemiring<SemiringT>(sr);
}

template <typename SemiringT>
MultiplicativeMonoidFromSemiring<SemiringT>
extractMul(SemiringT const &sr) {
    return MultiplicativeMonoidFromSemiring<SemiringT>(sr);
}

template <typename T_in1, typename T_out = T_in1>
        struct set_random {
        set_random() : seed_(0) {

            srand(seed_);
        }

        inline T_out operator()(T_in1 lhs) {
            return rand();
        }

        int seed_;
    };
}

#endif //GB_KUN_TYPES_HPP