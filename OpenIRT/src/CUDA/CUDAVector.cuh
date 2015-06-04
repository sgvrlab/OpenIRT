#ifndef CUDAVector_H
#define CUDAVector_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

#ifndef PI
#define PI   float(3.1415926535897932384626433832795)
#endif

struct __builtin_align__(8) cuVec2
{
    union __builtin_align__(8) { 
        struct __builtin_align__(8) { float x,y; };
        float e[2];
    };
    __device__ __host__ cuVec2() {}
    __device__ __host__ cuVec2(const float a, const float b) : x(a), y(b) {}
	__device__ __host__ cuVec2 operator+(const cuVec2 &v) const { return cuVec2(x+v.x,y+v.y); }
	__device__ __host__ cuVec2 operator-(const cuVec2 &v) const { return cuVec2(x-v.x,y-v.y); }
	__device__ __host__ const float &operator[](const unsigned int i) const { return e[i]; }
};

struct cuVec3
{
    union { 
        struct { float x,y,z; };
        float e[3];
    };
    __device__ __host__ cuVec3() {}
    __device__ __host__ cuVec3(const float a, const float b, const float c) : x(a), y(b), z(c) {}
	__device__ __host__ cuVec3(const float a) : x(a), y(a), z(a) {}
	__device__ __host__ cuVec3(const float4 &a) : x(a.x), y(a.y), z(a.z) {}
	__device__ __host__ inline void set(const cuVec3 &v) {*this = v;}
	__device__ __host__ inline void set(const float a, const float b, const float c) {x=a;y=b;z=c;}
    __device__ __host__ inline cuVec3 operator+(const cuVec3 &v) const { return cuVec3(x+v.x,y+v.y,z+v.z); }
    __device__ __host__ inline cuVec3 operator-(const cuVec3 &v) const { return cuVec3(x-v.x,y-v.y,z-v.z); }
    __device__ __host__ inline cuVec3 operator-() const { return cuVec3(-x,-y,-z); }
    __device__ __host__ inline cuVec3 operator*(const float d) const { return cuVec3(x*d,y*d,z*d); }
	__device__ __host__ inline cuVec3 operator*(const cuVec3 &v) const { return cuVec3(x*v.x,y*v.y,z*v.z); }
    __device__ __host__ inline cuVec3 operator/(const float d) const { return cuVec3(x/d,y/d,z/d); }
	__device__ __host__ inline cuVec3 operator/(const cuVec3 &v) const { return cuVec3(x/v.x,y/v.y,z/v.z); }
    __device__ __host__ inline const float &operator[](const unsigned int i) const { return e[i]; }
    __device__ __host__ inline float &operator[](const unsigned int i) { return e[i]; }
    __device__ __host__ inline cuVec3 cross(const cuVec3 &v) const { return cuVec3(y*v.z-z*v.y,z*v.x-x*v.z,x*v.y-y*v.x); }
    __device__ __host__ inline cuVec3 normalize() const { return *this * (1.f/sqrtf(magsqr())); }
	__device__ __host__ inline float length() const { return sqrtf(magsqr()); }
    __device__ __host__ inline float dot(const cuVec3 &v) const { return x*v.x+y*v.y+z*v.z; }
	__device__ __host__ inline float magsqr() const { return dot(*this); }
	__device__ __host__ inline cuVec3& operator+=(const cuVec3 &v){ x += v.x; y += v.y; z += v.z; return *this;}
	__device__ __host__ inline int indexOfMinComponent() const {return (e[0]< e[1] && e[0]< e[2]) ? 0 : (e[1] < e[2] ? 1 : 2);}
	__device__ __host__ inline int indexOfMaxComponent() const {return (e[0]> e[1] && e[0]> e[2]) ? 0 : (e[1] > e[2] ? 1 : 2);}
	__device__ __host__ inline float minComponent() const {return e[indexOfMinComponent()];}
	__device__ __host__ inline float maxComponent() const {return e[indexOfMaxComponent()];}
	__device__ __host__ inline int indexOfMaxAbsComponent() const {
		if (fabsf(e[0]) > fabsf(e[1]) && fabsf(e[0]) > fabsf(e[2]))
			return 0;
		else if (fabsf(e[1]) > fabsf(e[2]))
			return 1;
		else
			return 2;
	}


	__device__ __host__ cuVec3& operator=(float4 &v)
	{
		this->x = v.x;
		this->y = v.x;
		this->z = v.x;
		return *this;
	}

	__device__ __host__ bool operator == (const cuVec3 &v)
	{
		if(e[0] != v.e[0]) return false;
		if(e[1] != v.e[1]) return false;
		return e[2] == v.e[2];
	}

	__device__ __host__ bool operator != (const cuVec3 &v)
	{
		return !(*this == v);
	}

	__device__ __host__ void reset()
	{
		e[0] = e[1] = e[2] = 0.0f;
	}
};

__device__ __host__ inline cuVec2 operator*(const float t, const cuVec2 &v)
{
	return cuVec2(v.x*t, v.y*t);
}

__device__ __host__ inline cuVec3 operator*(const float t, const cuVec3 &v)
{
	return cuVec3(v.x*t, v.y*t, v.z*t);
}

__device__ __host__ inline float4 operator*(const float t, const float4 &v)
{
	return make_float4(v.x*t, v.y*t, v.z*t, v.w*t);
}

__device__ __host__ inline float4 operator*(const float4 &v, const float t)
{
	return make_float4(v.x*t, v.y*t, v.z*t, v.w*t);
}

__device__ __host__ inline float4 operator*(const float4 &v1, const float4 &v2)
{
	return make_float4(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w*v2.w);
}

__device__ __host__ inline float4 operator+(const float4 &v1, const float4 &v2)
{
	return make_float4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w);
}

__device__ __host__ inline float4 operator-(const float4 &v1, const float4 &v2)
{
	return make_float4(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, v1.w-v2.w);
}

__device__ __host__ inline float4 operator/(const float4 &v, const float t)
{
	return make_float4(v.x/t, v.y/t, v.z/t, v.w/t);
}

__device__ __host__ inline float4 operator-(const float4 &v)
{
	return make_float4(-v.x, -v.y, -v.z, v.w);
}


__device__ __host__ inline float4 operator+=(float4 &v1, const float4 &v2)
{
	v1 = make_float4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w);
	return v1;
}

__device__ __host__ inline float4 operator-=(float4 &v1, const float4 &v2)
{
	v1 = make_float4(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, v1.w-v2.w);
	return v1;
}

__device__ __host__ inline float4 operator+=(float4 &v1, const cuVec3 &v2)
{
	v1 = make_float4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w);
	return v1;
}

__device__ __host__ inline float4 operator-=(float4 &v1, const cuVec3 &v2)
{
	v1 = make_float4(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, v1.w);
	return v1;
}




__device__ __host__ inline float4 operator*(const cuVec3 &v1, const float4 &v2)
{
	return make_float4(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v2.w);
}

__device__ __host__ inline float4 operator*(const float4 &v1, const cuVec3 &v2)
{
	return make_float4(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w);
}

__device__ __host__ inline float4 operator+(const cuVec3 &v1, const float4 &v2)
{
	return make_float4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v2.w);
}

__device__ __host__ inline float4 operator+(const float4 &v1, const cuVec3 &v2)
{
	return make_float4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w);
}

__device__ __host__ inline float4 operator-(const cuVec3 &v1, const float4 &v2)
{
	return make_float4(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, -v2.w);
}

__device__ __host__ inline float4 operator-(const float4 &v1, const cuVec3 &v2)
{
	return make_float4(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, v1.w);
}

#endif // #ifndef VECTOR_GPU_H
