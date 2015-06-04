#ifndef COMMON_MATH_H
#define COMMON_MATH_H

// Useful constants
//
#define PI		3.141592f	// Pi
#define PI2		6.283185f	// Pi^2
#define HALF_PI	1.570796f	// Pi/2
#define E		2.718282f	// Euler's Number
#define DTOR	0.017452f	// Degree -> Radian
#define RTOD	57.29578f	// Radian -> Degree
#define EPSILON 0.00001f	// useful for float comparisons

// Useful macros
//
#define SQR(a)			((a)*(a))
#define SWAP(a,b)		{ a^=b; b^=a; a^=b; }
#define LERP(a,l,r)		((l) + (((r) - (l))*(a)))
#define CLAMP(v,l,r)	((v)<(l)?(l) : (v)>(r)?(r):v)

// Floating point comparisons with epsilon factored
// in, so that small fp instabilities won't matter
// (e.g. coordinate comparisons)

// a <= b 
#define FLOAT_LE(a,b) (((a)-(b))<EPSILON)
// a < b
#define FLOAT_LT(a,b) (((a)-(b))<-EPSILON)
// a >= b 
#define FLOAT_GE(a,b) (((b)-(a))<EPSILON)
// a > b 
#define FLOAT_GT(a,b) (((b)-(a))<-EPSILON)


#endif