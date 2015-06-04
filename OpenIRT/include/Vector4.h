#pragma once

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <iomanip>

#include <xmmintrin.h>

#include "mathdefines.h"
#include "Vector2.h"
#include "Vector3.h"

namespace irt
{

class Vector4  {
public:

	// Constructors
	Vector4() { e[0] = 0; e[1] = 0; e[2] = 0; e[3] = 0; }
	Vector4(float e0, float e1, float e2, float e3 = 0.0f) {e[0]=e0; e[1]=e1; e[2]=e2; e[3]=e3;}
	Vector4(float enew[4]) {e[0]=enew[0]; e[1]=enew[1]; e[2]=enew[2]; e[3]=enew[3];}
	inline Vector4(const Vector4& v) { e[0] = v.e[0]; e[1] = v.e[1]; e[2] = v.e[2]; e[3] = v.e[3]; }
	inline Vector4(const Vector3& v) { e[0] = v.e[0]; e[1] = v.e[1]; e[2] = v.e[2]; e[3] = 1.0f; }
//#ifdef NPR
	inline Vector4(const float v1, const Vector3& v2) { e[0] = v1; e[1] = v2.e[0]; e[2] = v2.e[1]; e[3] = v2.e[2]; }
	inline Vector4(const Vector2& v1, const Vector2& v2) { e[0] = v1.e[0]; e[1] = v1.e[1]; e[2] = v2.e[0]; e[3] = v2.e[1]; }
	inline Vector4(const Vector3& v1, const float& v2) { e[0] = v1.e[0]; e[1] = v1.e[1]; e[2] = v1.e[2]; e[3] = v2; }

//#endif

	// Member access: read
	float x() const { return e[0]; }
	float y() const { return e[1]; }
	float z() const { return e[2]; }
	float w() const { return e[3]; }

//#ifdef NPR
	Vector3 xyz() { return Vector3(e); }
	Vector2 xy() { return Vector2(e); }
	Vector2 zw() { return Vector2(e+2); }
//#endif

	// Member access: write
	void setX(float a) { e[0] = a; }
	void setY(float a) { e[1] = a; }
	void setXY(const Vector2& v) { e[0] = v.e[0]; e[1] = v.e[1];}
	void setZW(const Vector2& v) { e[2] = v.e[0]; e[3] = v.e[1];}
	void setZ(float a) { e[2] = a; }
	void setW(float a) { e[3] = a; }
	void set(float enew[4]) { e[0]=enew[0]; e[1]=enew[1]; e[2]=enew[2]; e[3]=enew[3]; }
	void set(float a, float b, float c, float d) { e[0]=a; e[1]=b; e[2]=c; e[3]=d; }
	void set(float enew) { e[0]=enew; e[1]=enew; e[2]=enew; e[3]=enew; }
	void set(const Vector3& v) { e[0]=v.e[0]; e[1]=v.e[1]; e[2]=v.e[2]; e[3]=1.f; }

	const Vector4& operator+() const { return *this; }
	Vector4 operator-() const { return Vector4(-e[0], -e[1], -e[2], -e[3]); }
	float& operator[](int i) { return e[i]; }
	float operator[](int i) const { return e[i];}

	Vector4& operator+=(const Vector4 &v2);
	Vector4& operator-=(const Vector4 &v2);

	Vector4& operator*=(const float t);
	Vector4& operator/=(const float t);

	Vector4& operator*=(const Vector4& v2);

	Vector4& operator*=(const float *t);
	Vector4& operator/=(const float *t);
	Vector4& operator+=(const float *t);
	Vector4& operator-=(const float *t);

	float length3() const { 
		return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); 
	}
	float length() const { 
		return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2] + e[3]*e[3]); 
	}
	float squaredLength3() const { 
		return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; 
	}
	float squaredLength() const { 		
		return e[0]*e[0] + e[1]*e[1] + e[2]*e[2] + e[3]*e[3]; 
	}

	void makeUnitVector();
	void makeUnitVector3();

	float minComponent() const { return e[indexOfMinComponent()]; }
	float maxComponent() const { return e[indexOfMaxComponent()]; }
	float maxAbsComponent() const { return e[indexOfMaxAbsComponent()]; }
	float minAbsComponent() const { return e[indexOfMinAbsComponent()]; }

	int indexOfMinComponent() const { 
		return (e[0]< e[1] && e[0]< e[2] && e[0] < e[3])? 0 : 
				((e[1] < e[2] && e[1] < e[3])? 1 : 
				 ((e[2] < e[3]) ? 2 : 3));
	}

	int indexOfMinAbsComponent() const {
		if (fabs(e[0]) < fabs(e[1]) && fabs(e[0]) < fabs(e[2]) && fabs(e[0]) < fabs(e[3]))
			return 0;
		if (fabs(e[1]) < fabs(e[2]) && fabs(e[1]) < fabs(e[3]))
			return 1;
		else if (fabs(e[2]) < fabs(e[3]))
			return 2;
		else
			return 3;
	}

	int indexOfMaxComponent() const {
		return (e[0]> e[1] && e[0]> e[2] && e[0] > e[3])? 0 : 
				((e[1] > e[2] && e[1] > e[3])? 1 : 
				((e[2] > e[3]) ? 2 : 3));
	}

	int indexOfMaxAbsComponent() const {
		if (fabs(e[0]) > fabs(e[1]) && fabs(e[0]) > fabs(e[2]) && fabs(e[0]) > fabs(e[3]))
			return 0;
		if (fabs(e[1]) > fabs(e[2]) && fabs(e[1]) > fabs(e[3]))
			return 1;
		else if (fabs(e[2]) > fabs(e[3]))
			return 2;
		else
			return 3;
	}

	//__declspec(align(16)) float e[4];
	union {
		float e[4];
		__m128 e4;
	};
};


inline bool operator==(const Vector4 &t1, const Vector4 &t2) {
	return ((t1[0]==t2[0])&&(t1[1]==t2[1])&&(t1[2]==t2[2])&&(t1[3]==t2[3]));
}

inline bool operator!=(const Vector4 &t1, const Vector4 &t2) {
	return ((t1[0]!=t2[0])||(t1[1]!=t2[1])||(t1[2]!=t2[2])||(t1[3]!=t2[3]));
}

inline std::istream &operator>>(std::istream &is, Vector4 &t) {
	is >> t[0] >> t[1] >> t[2] >> t[3];
	return is;
}

inline std::ostream &operator<<(std::ostream &os, const Vector4 &t) {
	os << t[0] << " " << t[1] << " " << t[2] << " " << t[3];
	return os;
}

inline Vector4 unitVector(const Vector4& v) {
	float k = 1.0f / sqrt(v.e[0]*v.e[0] + v.e[1]*v.e[1] + v.e[2]*v.e[2] + v.e[3]*v.e[3]);
	return Vector4(v.e[0]*k, v.e[1]*k, v.e[2]*k, v.e[3]*k);
}

inline Vector3 unitVector3(const Vector4& v) {
	float k = 1.0f / sqrt(v.e[0]*v.e[0] + v.e[1]*v.e[1] + v.e[2]*v.e[2]);
	return Vector3(v.e[0]*k, v.e[1]*k, v.e[2]*k);
}

inline void Vector4::makeUnitVector() {
	float k = 1.0f / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2] + e[3]*e[3]);
	*this *= k;	
}

inline void Vector4::makeUnitVector3() {
	float k = 1.0f / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
	e[0] *= k; e[1] *= k; e[2] *= k;
}

inline Vector4 operator+(const Vector4 &v1, const Vector4 &v2) {	
	return Vector4( v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2], v1.e[3] + v2.e[3]);
}

inline Vector4 operator+(const Vector4 &v1, const float v2) {	
	return Vector4( v1.e[0] + v2, v1.e[1] + v2, v1.e[2] + v2, v1.e[3] + v2);
}


inline Vector4 operator-(const Vector4 &v1, const Vector4 &v2) {
	return Vector4( v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2], v1.e[3] - v2.e[3]);
}

inline Vector4 operator-(const float v1, const Vector4 &v2) {
	return Vector4( v1 - v2.e[0], v1 - v2.e[1], v1 - v2.e[2], v1 - v2.e[3]);
}

inline Vector4 operator*(const float t, const Vector4 &v) {
	return Vector4(t*v.e[0], t*v.e[1], t*v.e[2], t*v.e[3]); 
}

inline Vector4 operator*(const float *t, const Vector4 &v) {
	return Vector4(t[0]*v.e[0], t[1]*v.e[1], t[2]*v.e[2], t[3]*v.e[3]); 
}

inline Vector4 operator*(const Vector4 &v, float t) {
	return Vector4(t*v.e[0], t*v.e[1], t*v.e[2], t*v.e[3]); 
}

inline Vector4 operator*(const Vector4 &v, const float *t) {
	return Vector4(t[0]*v.e[0], t[1]*v.e[1], t[2]*v.e[2], t[3]*v.e[3]); 
}

inline Vector4 operator/(const Vector4 &v, float t) {
	return Vector4(v.e[0]/t, v.e[1]/t, v.e[2]/t, v.e[3]/t); 
}

inline Vector4 operator/(const Vector4 &v, const float *t) {
	return Vector4(v.e[0]/t[0], v.e[1]/t[1], v.e[2]/t[2], v.e[3]/t[3]); 
}

inline float dot(const Vector4 &v1, const Vector4 &v2) {
	return v1.e[0]*v2.e[0] + v1.e[1]*v2.e[1]  + v1.e[2]*v2.e[2] + v1.e[3]*v2.e[3];
}

inline float dot(const Vector4 &v1, const float *v2) {
	return v1.e[0]*v2[0] + v1.e[1]*v2[1]  + v1.e[2]*v2[2] + v1.e[3]*v2[3];
}

inline float dot4(const float *v1, const float *v2) {
	return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3];
}

inline Vector4 cross(const Vector4 &v1, const Vector4 &v2) {
	return Vector4( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
					(v1.e[2]*v2.e[0] - v1.e[0]*v2.e[2]),
					(v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}

inline bool operator<(const Vector4 &v1, const Vector4 &v2) {
	return (FLOAT_LT(v1.e[0], v2.e[0]) && FLOAT_LT(v1.e[1], v2.e[1]) && FLOAT_LT(v1.e[2], v2.e[2]) && FLOAT_LT(v1.e[3], v2.e[3]));
}

inline bool operator<=(const Vector4 &v1, const Vector4 &v2) {
	return (FLOAT_LE(v1.e[0], v2.e[0]) && FLOAT_LE(v1.e[1], v2.e[1]) && FLOAT_LE(v1.e[2], v2.e[2]) && FLOAT_LE(v1.e[3], v2.e[3]));
}

inline bool operator>(const Vector4 &v1, const Vector4 &v2) {
	return (FLOAT_GT(v1.e[0], v2.e[0]) && FLOAT_GT(v1.e[1], v2.e[1]) && FLOAT_GT(v1.e[2], v2.e[2]) && FLOAT_GT(v1.e[3], v2.e[3]));
}

inline bool operator>=(const Vector4 &v1, const Vector4 &v2) {
	return (FLOAT_GE(v1.e[0], v2.e[0]) && FLOAT_GE(v1.e[1], v2.e[1]) && FLOAT_GE(v1.e[2], v2.e[2]) && FLOAT_GE(v1.e[3], v2.e[3]));	
}

inline Vector4& Vector4::operator+=(const Vector4 &v){
	_mm_store_ps(this->e, _mm_add_ps(_mm_load_ps(this->e),_mm_load_ps(v.e)));	
	return *this;
}

inline Vector4& Vector4::operator-=(const Vector4& v) {
	_mm_store_ps(this->e, _mm_sub_ps(_mm_load_ps(this->e),_mm_load_ps(v.e)));	
	return *this;
}
inline Vector4& Vector4::operator*=(const Vector4& v2) {	
	e[0] = e[0]*v2.e[0];
	e[1] = e[1]*v2.e[1];
	e[2] = e[2]*v2.e[2];
	e[3] = e[3]*v2.e[3];
	
	return *this;
}

inline Vector4& Vector4::operator*=(const float t) {
	_mm_store_ps(this->e, _mm_mul_ps(_mm_load_ps(this->e),_mm_load1_ps(&t)));	
	return *this;
}

inline Vector4& Vector4::operator/=(const float t) {
	_mm_store_ps(this->e, _mm_div_ps(_mm_load_ps(this->e),_mm_load1_ps(&t)));	
	return *this;
}

/**
 * Piecewise multiply with array of floats. t needs
 * to be 16-byte aligned !
 */
inline Vector4& Vector4::operator*=(const float *t) {
	_mm_store_ps(this->e, _mm_mul_ps(_mm_load_ps(this->e),_mm_load_ps(t)));	
	return *this;
}

/**
* Piecewise addition with array of floats. t needs
* to be 16-byte aligned !
*/
inline Vector4& Vector4::operator+=(const float *t) {
	_mm_store_ps(this->e, _mm_add_ps(_mm_load_ps(this->e),_mm_load_ps(t)));	
	return *this;
}

/**
* Piecewise subtraction with array of floats. t needs
* to be 16-byte aligned !
*/
inline Vector4& Vector4::operator-=(const float *t) {
	_mm_store_ps(this->e, _mm_sub_ps(_mm_load_ps(this->e),_mm_load_ps(t)));	
	return *this;
}
/**
* Piecewise division with array of floats. t needs
* to be 16-byte aligned !
*/
inline Vector4& Vector4::operator/=(const float *t) {
	_mm_store_ps(this->e, _mm_div_ps(_mm_load_ps(this->e),_mm_load_ps(t)));	
	return *this;
}

typedef union { float v[4]; __m128 v4; } SIMDVec4;

inline Vector4 toLH(const Vector4& v) {
	Vector4 out(v);
	out.e[2] = -v.e[2];
	return out;
}

inline Vector4 clamp(const Vector4& v, float low, float high) {
	Vector4 out;
	out.e[0] = CLAMP(v.e[0],low,high);
	out.e[1] = CLAMP(v.e[1],low,high);
	out.e[2] = CLAMP(v.e[2],low,high);
	out.e[3] = CLAMP(v.e[3],low,high);

	return out;
}

inline Vector4 saturate(const Vector4& v) {
	Vector4 out;
	out.e[0] = SATURATE(v.e[0]);
	out.e[1] = SATURATE(v.e[1]);
	out.e[2] = SATURATE(v.e[2]);
	out.e[3] = SATURATE(v.e[3]);

	return out;
}

};