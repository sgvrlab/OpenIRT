#ifndef CUDAVector_H
#define CUDAVector_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

struct __builtin_align__(8) cuVec2
{
    union __builtin_align__(8) { 
        struct __builtin_align__(8) { float x,y; };
        float e[2];
    };
    __device__ cuVec2() {}
    __device__ cuVec2(const float a, const float b) : x(a), y(b) {}
	__device__ cuVec2 operator+(const cuVec2 &v) const { return cuVec2(x+v.x,y+v.y); }
	__device__ cuVec2 operator-(const cuVec2 &v) const { return cuVec2(x-v.x,y-v.y); }
	__device__ const float &operator[](const unsigned int i) const { return e[i]; }
};

struct cuVec3
{
    union { 
        struct { float x,y,z; };
        float e[3];
    };
    __device__ cuVec3() {}
    __device__ cuVec3(const float a, const float b, const float c) : x(a), y(b), z(c) {}
    __device__ inline cuVec3 operator+(const cuVec3 &v) const { return cuVec3(x+v.x,y+v.y,z+v.z); }
    __device__ inline cuVec3 operator-(const cuVec3 &v) const { return cuVec3(x-v.x,y-v.y,z-v.z); }
    __device__ inline cuVec3 operator-() const { return cuVec3(-x,-y,-z); }
    __device__ inline cuVec3 operator*(const float d) const { return cuVec3(x*d,y*d,z*d); }
	__device__ inline cuVec3 operator*(const cuVec3 &v) const { return cuVec3(x*v.x,y*v.y,z*v.z); }
    __device__ inline const float &operator[](const unsigned int i) const { return e[i]; }
    __device__ inline float &operator[](const unsigned int i) { return e[i]; }
    __device__ inline cuVec3 cross(const cuVec3 &v) const { return cuVec3(y*v.z-z*v.y,z*v.x-x*v.z,x*v.y-y*v.x); }
    __device__ inline cuVec3 normalize() const { return *this * (1.f/sqrtf(magsqr())); }
	__device__ inline float length() const { return sqrtf(magsqr()); }
    __device__ inline float dot(const cuVec3 &v) const { return x*v.x+y*v.y+z*v.z; }
	__device__ inline float magsqr() const { return dot(*this); }
	__device__ inline cuVec3& operator+=(const cuVec3 &v){ x += v.x; y += v.y; z += v.z; return *this;}
};

__device__ inline cuVec2 operator*(const float t, const cuVec2 &v)
{
	return cuVec2(v.x*t, v.y*t);
}

__device__ inline cuVec3 operator*(const float t, const cuVec3 &v)
{
	return cuVec3(v.x*t, v.y*t, v.z*t);
}

#endif // #ifndef VECTOR_GPU_H
