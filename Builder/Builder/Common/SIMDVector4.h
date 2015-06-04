#ifndef COMMON_SIMDVECTOR4_H
#define COMMON_SIMDVECTOR4_H

/********************************************************************
	created:	2004/10/17
	created:	17:10:2004   15:52
	filename: 	C:\MSDev\MyProjects\Renderer\Common\SIMDVector4.h
	file path:	C:\MSDev\MyProjects\Renderer\Common
	file base:	SIMDVector4
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Class encapsulating 4 Vectors for SIMD use with SSE or
				similar instructions
*********************************************************************/

/*
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <memory.h>
#include <xmmintrin.h>

#include "Math.h"

class SIMDVector4  {
public:

	SIMDVector4() { }

	SIMDVector4(Vector4 &e1, Vector4 &e2, Vector4 &e3) {
		e[0]=e1; e[1]=e2; e[2]=e3;
	}
	SIMDVector4(Vector4 &e1, Vector4 &e2, Vector4 &e3, Vector4 &e4) {
		e[0]=e1; e[1]=e2; e[2]=e3; e[2]=e4;
	}

	SIMDVector4(Vector4 enew[4]) {
		e[0]=enew[0]; e[1]=enew[1]; e[2]=enew[2]; e[3]=enew[3];
	}

	inline SIMDVector4(const SIMDVector4 &v) {
		e[0] = v.e[0]; e[1] = v.e[1]; e[2] = v.e[2]; e[3] = v.e[3];
	}

	float x() const { return e[0]; }
	float y() const { return e[1]; }
	float z() const { return e[2]; }
	void setX(float a) { e[0] = a; }
	void setY(float a) { e[1] = a; }
	void setZ(float a) { e[2] = a; }
	void set(float enew[3]) {e[0]=enew[0]; e[1]=enew[1]; e[2]=enew[2]; }

	const SIMDVector4& operator+() const { return *this; }
	SIMDVector4 operator-() const { return SIMDVector4(-e[0], -e[1], -e[2]); }
	float& operator[](int i) {  return e[i]; }
	float operator[](int i) const { return e[i];}

	SIMDVector4& operator+=(const SIMDVector4 &v2);
	SIMDVector4& operator-=(const SIMDVector4 &v2);
	SIMDVector4& operator*=(const float t);
	SIMDVector4& operator/=(const float t);

	float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
	float squaredLength() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }

	void makeUnitVector();

	float minComponent() const { return e[indexOfMinComponent()]; }
	float maxComponent() const { return e[indexOfMaxComponent()]; }
	float maxAbsComponent() const { return e[indexOfMaxAbsComponent()]; }
	float minAbsComponent() const { return e[indexOfMinAbsComponent()]; }
	int indexOfMinComponent() const { 
		return (e[0]< e[1] && e[0]< e[2]) ? 0 : (e[1] < e[2] ? 1 : 2);
	}

	int indexOfMinAbsComponent() const {
		if (fabs(e[0]) < fabs(e[1]) && fabs(e[0]) < fabs(e[2]))
			return 0;
		else if (fabs(e[1]) < fabs(e[2]))
			return 1;
		else
			return 2;
	}

	int indexOfMaxComponent() const {
		return (e[0]> e[1] && e[0]> e[2]) ? 0 : (e[1] > e[2] ? 1 : 2);
	}

	int indexOfMaxAbsComponent() const {
		if (fabs(e[0]) > fabs(e[1]) && fabs(e[0]) > fabs(e[2]))
			return 0;
		else if (fabs(e[1]) > fabs(e[2]))
			return 1;
		else
			return 2;
	}

	__declspec(align(16)) Vector4 e[4];
};


inline bool operator==(const SIMDVector4 &t1, const SIMDVector4 &t2) {
	return ((t1[0]==t2[0])&&(t1[1]==t2[1])&&(t1[2]==t2[2]));
}

inline bool operator!=(const SIMDVector4 &t1, const SIMDVector4 &t2) {
	return ((t1[0]!=t2[0])||(t1[1]!=t2[1])||(t1[2]!=t2[2]));
}

inline istream &operator>>(istream &is, SIMDVector4 &t) {
	is >> t[0] >> t[1] >> t[2];
	return is;
}

inline ostream &operator<<(ostream &os, const SIMDVector4 &t) {
	os << t[0] << " " << t[1] << " " << t[2];
	return os;
}

inline SIMDVector4 unitVector(const SIMDVector4& v) {
	float k = 1.0f / sqrt(v.e[0]*v.e[0] + v.e[1]*v.e[1] + v.e[2]*v.e[2]);
	return SIMDVector4(v.e[0]*k, v.e[1]*k, v.e[2]*k);
}

inline void SIMDVector4::makeUnitVector() {
	float k = 1.0f / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
	e[0] *= k; e[1] *= k; e[2] *= k;
}

inline SIMDVector4 operator+(const SIMDVector4 &v1, const SIMDVector4 &v2) {
	return SIMDVector4( v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline SIMDVector4 operator-(const SIMDVector4 &v1, const SIMDVector4 &v2) {
	return SIMDVector4( v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline SIMDVector4 operator*(float t, const SIMDVector4 &v) {
	return SIMDVector4(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

inline SIMDVector4 operator*(const SIMDVector4 &v, float t) {
	return SIMDVector4(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

inline SIMDVector4 operator/(const SIMDVector4 &v, float t) {
	return SIMDVector4(v.e[0]/t, v.e[1]/t, v.e[2]/t); 
}


inline float dot(const SIMDVector4 &v1, const SIMDVector4 &v2) {
	return v1.e[0]*v2.e[0] + v1.e[1]*v2.e[1]  + v1.e[2]*v2.e[2];
}

inline float dot(const SIMDVector4 &v1, const float *v2) {
	return v1.e[0]*v2[0] + v1.e[1]*v2[1]  + v1.e[2]*v2[2];
}

inline float dot(const float *v1, const float *v2) {
	return v1[0]*v2[0] + v1[1]*v2[1]  + v1[2]*v2[2];
}

inline SIMDVector4 cross(const SIMDVector4 &v1, const SIMDVector4 &v2) {
	return SIMDVector4( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
		(v1.e[2]*v2.e[0] - v1.e[0]*v2.e[2]),
		(v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}

inline bool operator<(const SIMDVector4 &v1, const SIMDVector4 &v2) {
	return (FLOAT_LT(v1.e[0], v2.e[0]) && FLOAT_LT(v1.e[1], v2.e[1]) && FLOAT_LT(v1.e[2], v2.e[2]));
}

inline bool operator<=(const SIMDVector4 &v1, const SIMDVector4 &v2) {
	return (FLOAT_LE(v1.e[0], v2.e[0]) && FLOAT_LE(v1.e[1], v2.e[1]) && FLOAT_LE(v1.e[2], v2.e[2]));
}

inline bool operator>(const SIMDVector4 &v1, const SIMDVector4 &v2) {
	return (FLOAT_GT(v1.e[0], v2.e[0]) && FLOAT_GT(v1.e[1], v2.e[1]) && FLOAT_GT(v1.e[2], v2.e[2]));
}

inline bool operator>=(const SIMDVector4 &v1, const SIMDVector4 &v2) {
	return (FLOAT_GE(v1.e[0], v2.e[0]) && FLOAT_GE(v1.e[1], v2.e[1]) && FLOAT_GE(v1.e[2], v2.e[2]));	
}

inline SIMDVector4& SIMDVector4::operator+=(const SIMDVector4 &v){
	e[0]  += v.e[0];
	e[1]  += v.e[1];
	e[2]  += v.e[2];
	return *this;
}

inline SIMDVector4& SIMDVector4::operator-=(const SIMDVector4& v) {
	e[0]  -= v.e[0];
	e[1]  -= v.e[1];
	e[2]  -= v.e[2];
	return *this;
}

inline SIMDVector4& SIMDVector4::operator*=(const float t) {
	e[0]  *= t;
	e[1]  *= t;
	e[2]  *= t;
	return *this;
}

inline SIMDVector4& SIMDVector4::operator/=(const float t) {
	e[0]  /= t;
	e[1]  /= t;
	e[2]  /= t;
	return *this;
}

inline
SIMDVector4 reflect(const SIMDVector4& in, const SIMDVector4& normal)
{
	// assumes unit length normal
	return in - normal * (2 * dot(in, normal));
}
*/

#endif
