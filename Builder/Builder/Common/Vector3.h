#ifndef COMMON_VECTOR3_H
#define COMMON_VECTOR3_H

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <iomanip>

//#include "Math.h"
#include "mathdefines.h"

class Vector3  {
public:
    
    Vector3() { e[0] = 0; e[1] = 0; e[2] = 0;}
    Vector3(float e0, float e1, float e2) {e[0]=e0; e[1]=e1; e[2]=e2;}
	Vector3(float enew[3]) {e[0]=enew[0]; e[1]=enew[1]; e[2]=enew[2];}
    float x() const { return e[0]; }
    float y() const { return e[1]; }
    float z() const { return e[2]; }
    void setX(const float a) { e[0] = a; }
    void setY(const float a) { e[1] = a; }
    void setZ(const float a) { e[2] = a; }
	void set(const float *enew) {e[0]=enew[0]; e[1]=enew[1]; e[2]=enew[2]; }

    inline Vector3(const Vector3 &v) {
         e[0] = v.e[0]; e[1] = v.e[1]; e[2] = v.e[2];
    }

    const Vector3& operator+() const { return *this; }
    Vector3 operator-() const { return Vector3(-e[0], -e[1], -e[2]); }
    float& operator[](int i) {  return e[i]; }
    float operator[](int i) const { return e[i];}

    Vector3& operator+=(const Vector3 &v2);
    Vector3& operator-=(const Vector3 &v2);
    Vector3& operator*=(const float t);
    Vector3& operator/=(const float t);
    Vector3& operator/=(const Vector3 &v2);

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

    float e[3];
};

// sungeui start -----------
// make _Vector4 for out-of-core accessing vector3 data and 
// this is just temporary solution; this class is just for minimizing coding effort

class _Vector4: public Vector3
{
	float m_alpha;			// it's dummy now

public:
	_Vector4(Vector3 &v) {
		e[0] = v.e[0];
		e[1] = v.e[1];
		e[2] = v.e[2];
	}

	_Vector4() {
		e[0] = .0f;
		e[1] = .0f;
		e[2] = .0f;
	}
};
#define CONVERT_V3_to_V4(s, d) {d.e[0]=s.e[0]; d.e[1]=s.e[1]; d.e[2]=s.e[2];}
#define CONVERT_V4_to_V3(s, d) {d.e[0]=s.e[0]; d.e[1]=s.e[1]; d.e[2]=s.e[2];}
// sungeui end -----------------


inline bool operator==(const Vector3 &t1, const Vector3 &t2) {
   return ((t1[0]==t2[0])&&(t1[1]==t2[1])&&(t1[2]==t2[2]));
}

inline bool operator!=(const Vector3 &t1, const Vector3 &t2) {
   return ((t1[0]!=t2[0])||(t1[1]!=t2[1])||(t1[2]!=t2[2]));
}

inline std::istream &operator>>(std::istream &is, Vector3 &t) {
   is >> t[0] >> t[1] >> t[2];
   return is;
}

inline std::ostream &operator<<(std::ostream &os, const Vector3 &t) {
   os << t[0] << " " << t[1] << " " << t[2];
   return os;
}

inline Vector3 unitVector(const Vector3& v) {
    float k = 1.0f / sqrt(v.e[0]*v.e[0] + v.e[1]*v.e[1] + v.e[2]*v.e[2]);
    return Vector3(v.e[0]*k, v.e[1]*k, v.e[2]*k);
}

inline void Vector3::makeUnitVector() {
    float k = 1.0f / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

inline Vector3 operator+(const Vector3 &v1, const Vector3 &v2) {
    return Vector3( v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline Vector3 operator-(const Vector3 &v1, const Vector3 &v2) {
    return Vector3( v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline Vector3 operator*(float t, const Vector3 &v) {
    return Vector3(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

inline Vector3 operator*(const Vector3 &v, float t) {
    return Vector3(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

inline Vector3 operator/(const Vector3 &v, float t) {
    return Vector3(v.e[0]/t, v.e[1]/t, v.e[2]/t); 
}

inline Vector3 operator/(const Vector3 &v1, const Vector3 &v2) {
    return Vector3(v1.e[0]/v2.e[0], v1.e[1]/v2.e[1], v1.e[2]/v2.e[2]); 
}

inline float dot(const Vector3 &v1, const Vector3 &v2) {
    return v1.e[0]*v2.e[0] + v1.e[1]*v2.e[1]  + v1.e[2]*v2.e[2];
}

inline float dot(const Vector3 &v1, const float *v2) {
	return v1.e[0]*v2[0] + v1.e[1]*v2[1]  + v1.e[2]*v2[2];
}

inline float dot(const float *v1, const float *v2) {
	return v1[0]*v2[0] + v1[1]*v2[1]  + v1[2]*v2[2];
}

inline Vector3 cross(const Vector3 &v1, const Vector3 &v2) {
    return Vector3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                      (v1.e[2]*v2.e[0] - v1.e[0]*v2.e[2]),
                      (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}

inline Vector3 vmul(const Vector3 &v1, const Vector3 &v2) {
	return Vector3( v1.e[0]*v2.e[0], v1.e[1]*v2.e[1], v1.e[2]*v2.e[2]);
}

inline bool operator<(const Vector3 &v1, const Vector3 &v2) {
	return (FLOAT_LT(v1.e[0], v2.e[0]) && FLOAT_LT(v1.e[1], v2.e[1]) && FLOAT_LT(v1.e[2], v2.e[2]));
}

inline bool operator<=(const Vector3 &v1, const Vector3 &v2) {
	return (FLOAT_LE(v1.e[0], v2.e[0]) && FLOAT_LE(v1.e[1], v2.e[1]) && FLOAT_LE(v1.e[2], v2.e[2]));
}

inline bool operator>(const Vector3 &v1, const Vector3 &v2) {
	return (FLOAT_GT(v1.e[0], v2.e[0]) && FLOAT_GT(v1.e[1], v2.e[1]) && FLOAT_GT(v1.e[2], v2.e[2]));
}

inline bool operator>=(const Vector3 &v1, const Vector3 &v2) {
	return (FLOAT_GE(v1.e[0], v2.e[0]) && FLOAT_GE(v1.e[1], v2.e[1]) && FLOAT_GE(v1.e[2], v2.e[2]));	
}

inline Vector3& Vector3::operator+=(const Vector3 &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

inline Vector3& Vector3::operator-=(const Vector3& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

inline Vector3& Vector3::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

inline Vector3& Vector3::operator/=(const float t) {
    e[0]  /= t;
    e[1]  /= t;
    e[2]  /= t;
    return *this;
}

inline Vector3& Vector3::operator/=(const Vector3& v) {
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

inline
Vector3 reflect(const Vector3& in, const Vector3& normal)
{
  // assumes unit length normal
  return in - normal * (2 * dot(in, normal));
}

// Area of a triangle:
inline float triangleArea(Vector3 v1, Vector3 v2, Vector3 v3) {
	return (cross(v1,v2) + cross(v2,v3) + cross(v3,v1)).length() * 0.5f;
}

typedef std::vector<_Vector4> Vector3List;	
typedef Vector3List::iterator Vector3ListIterator;


#endif
