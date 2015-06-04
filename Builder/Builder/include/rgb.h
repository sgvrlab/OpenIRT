#ifndef RGB_H
#define RGB_H

#include <math.h>

/********************************************************************
	created:	2004/06/08
	created:	8.6.2004   0:38
	filename: 	c:\MSDev\MyProjects\Renderer\Common\rgb.h
	file base:	rgb
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Simple RGB color wrapper class
*********************************************************************/


#include <assert.h>
#include <iostream>
#include <ostream>

class rgb  {
public:

	rgb() {}
	rgb(float r, float g, float b) { data[0] = r; data[1] = g; data[2] = b; }
	rgb(const rgb &c) 
	{ data[0] = c.r(); data[1] = c.g(); data[2] = c.b(); }
	float r() const { return data[0]; }
	float g() const { return data[1]; }
	float b() const { return data[2]; }
	float sum() const { return data[0] + data[1] + data[2]; }
	float sumabs() const { return fabs(data[0]) + fabs(data[1]) + fabs(data[2]); }

	void clamp() {
		/*
		data[0] = max(min(data[0], 1.0f), 0.0f);
		data[1] = max(min(data[1], 1.0f), 0.0f);
		data[2] = max(min(data[2], 1.0f), 0.0f);		
		*/
	}

	rgb operator+() const { return rgb( data[0], data[1], data[2]); }
	rgb operator-() const { return rgb(-data[0],-data[1],-data[2]); }
	float operator[](int i) const {assert(i >= 0 && i < 3); return data[i];}
	float& operator[](int i) {assert(i >= 0 && i < 3); return data[i];} 

	rgb& operator+=(const rgb &c) { data[0] += c[0]; data[1] += c[1];
	data[2] += c[2]; return *this; } 
	rgb& operator-=(const rgb &c) { data[0] -= c[0]; data[1] -= c[1];
	data[2] -= c[2]; return *this; } 
	rgb& operator*=(const rgb &c) { data[0] *= c[0]; data[1] *= c[1];
	data[2] *= c[2]; return *this; } 
	rgb& operator/=(const rgb &c) { data[0] /= c[0]; data[1] /= c[1];
	data[2] /= c[2]; return *this; } 
	rgb& operator*=(float f) { data[0] *= f; data[1] *= f;
	data[2] *= f; return *this; } 
	rgb& operator/=(float f) { data[0] /= f; data[1] /= f;
	data[2] /= f; return *this; } 

	float data[3];
};

inline bool operator==(rgb c1, rgb c2) {
	return (c1.r() == c2.r() && c1.g() == c2.g() && c1.b() == c2.b()); }
inline bool operator!=(rgb c1, rgb c2) {
	return (c1.r() != c2.r() || c1.g() != c2.g() || c1.b() != c2.b()); }

inline bool operator<(rgb c1, rgb c2) {
	return (c1.sumabs() < c2.sumabs()); }
inline bool operator<=(rgb c1, rgb c2) {
	return (c1.sumabs() <= c2.sumabs()); }
inline bool operator>(rgb c1, rgb c2) {
	return (c1.sumabs() > c2.sumabs()); }
inline bool operator>=(rgb c1, rgb c2) {
	return (c1.sumabs() >= c2.sumabs()); }

inline rgb operator+(rgb c1, rgb c2) {
	return rgb(c1.r()+c2.r(), c1.g()+c2.g(), c1.b()+c2.b());}
inline rgb operator-(rgb c1, rgb c2) {
	return rgb(c1.r()-c2.r(), c1.g()-c2.g(), c1.b()-c2.b());}
inline rgb operator*(rgb c1, rgb c2) {
	return rgb(c1.r()*c2.r(), c1.g()*c2.g(), c1.b()*c2.b());}
inline rgb operator/(rgb c1, rgb c2) {
	return rgb(c1.r()/c2.r(), c1.g()/c2.g(), c1.b()/c2.b());}
inline rgb operator*(rgb c, float f) {
	return rgb(c.r()*f, c.g()*f, c.b()*f);}
inline rgb operator*(float f, rgb c) {
	return rgb(c.r()*f, c.g()*f, c.b()*f);}
inline rgb operator/(rgb c, float f) {
	return rgb(c.r()/f, c.g()/f, c.b()/f);}

inline std::ostream &operator<<(std::ostream &os, const rgb &t) {
	os << t[0] << " " << t[1] << " " << t[2];
	return os;
}

class rgba : public rgb {
public:
	float alpha;
	rgba() {}
	rgba(float r, float g, float b, float a) { data[0] = r; data[1] = g; data[2] = b; alpha = a;}
};

#endif