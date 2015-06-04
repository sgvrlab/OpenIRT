#pragma once

#include "Vector3.h"
#include "Vector4.h"
#include <iostream>
#include <iomanip>
#include <math.h>

namespace irt
{

class Matrix
{
public:
	Matrix() {}
	Matrix(const Matrix & orig);

	void invert();
	void transpose();
	Matrix getInverse()const;
	Matrix getTranspose()const;
	float &getRef(int i, int j) {return x[i][j];}
	float &getRef(int i) {return x[i/4][i%4];}

	Matrix & operator+= (const Matrix& right_op);
	Matrix & operator-= (const Matrix& right_op);
	Matrix & operator*= (const Matrix& right_op);
	Matrix & operator*= (float right_op);

	friend Matrix operator- (const Matrix& left_op, const Matrix& right_op);
	friend Matrix operator+ (const Matrix& left_op, const Matrix& right_op);
	friend Matrix operator* (const Matrix& left_op, const Matrix& right_op);

	// matrix mult. performed left to right
	friend Vector3 operator* (const Matrix& left_op, const Vector3& right_op);
	friend Vector4 operator* (const Matrix& left_op, const Vector4& right_op);
	// transform vector by a matrix
	friend Matrix operator* (const Matrix & left_op, float right_op);

	friend bool operator== (const Matrix& left_op, const Matrix& right_op);
	friend bool operator!= (const Matrix& left_op, const Matrix& right_op);

	friend Vector3 transformLoc(const Matrix& left_op, const Vector3& right_op); 
	friend Vector3 transformVec(const Matrix& left_op, const Vector3& right_op); 

	friend Matrix zeroMatrix();
	friend Matrix identityMatrix();
	friend Matrix translate(float _x, float _y, float _z);
	friend Matrix scale(float _x, float _y, float _z);
	friend Matrix rotate(const Vector3 & axis, float angle);
	friend Matrix rotateX(float angle);    //
	friend Matrix rotateY(float angle);    // More efficient than arbitrary axis
	friend Matrix rotateZ(float angle);    //
	friend Matrix viewMatrix(const Vector3& eye, const Vector3 & gaze, 
							 const Vector3& up);
	friend std::ostream & operator<< (std::ostream& out, const Matrix& right_op);

	float determinant(); 
	float x[4][4];
};

// 3x3 matrix determinant -- helper function 
inline float det3 (float a, float b, float c, 
				   float d, float e, float f, 
				   float g, float h, float i)
{ return a*e*i + d*h*c + g*b*f - g*e*c - d*b*i - a*h*f; }

// cheap way of defining a transform on the ray.
/// TODO: do this by matrix multiplication?
typedef Vector3 RayTransform;
extern RayTransform identityTransform;

};