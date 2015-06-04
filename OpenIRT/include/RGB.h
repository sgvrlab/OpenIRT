#ifndef COMMON_RGB_H
#define COMMON_RGB_H

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include <float.h>

class RGBf
{
public:
	RGBf() {e[0] = e[1] = e[2] = 0.0f;}
	RGBf(float r, float g, float b) {e[0] = r; e[1] = g; e[2] = b;}
	RGBf(float c) {e[0] = e[1] = e[2] = c;}
	RGBf(float *c) {e[0] = c[0]; e[1] = c[1]; e[2] = c[2];}

	float r() const {return e[0];}
	float g() const {return e[1];}
	float b() const {return e[2];}

	void set(float r, float g, float b) {e[0] = r; e[1] = g; e[2] = b;}
	void setR(float r) {e[0] = r;}
	void setG(float g) {e[1] = g;}
	void setB(float b) {e[2] = b;}

	float sum() const {return e[0] + e[1] + e[2];}
	float sumabs() const {return fabs(e[0]) + fabs(e[1]) + fabs(e[2]);}

    const RGBf& operator+() const { return *this; }
    RGBf operator-() const { return RGBf(-e[0], -e[1], -e[2]); }

	RGBf& operator+=(const RGBf &v2);
    RGBf& operator-=(const RGBf &v2);
    RGBf& operator*=(const float t);
    RGBf& operator/=(const float t);
    RGBf& operator/=(const RGBf &v2);

public:
	float e[3];
};

inline bool operator==(const RGBf& c1, const RGBf& c2) {
	return (c1.r() == c2.r() && c1.g() == c2.g() && c1.b() == c2.b()); }
inline bool operator!=(const RGBf& c1, const RGBf& c2) {
	return (c1.r() != c2.r() || c1.g() != c2.g() || c1.b() != c2.b()); }

inline bool operator<(const RGBf& c1, const RGBf& c2) {
	return (c1.sumabs() < c2.sumabs()); }
inline bool operator<=(const RGBf& c1, const RGBf& c2) {
	return (c1.sumabs() <= c2.sumabs()); }
inline bool operator>(const RGBf& c1, const RGBf& c2) {
	return (c1.sumabs() > c2.sumabs()); }
inline bool operator>=(const RGBf& c1, const RGBf& c2) {
	return (c1.sumabs() >= c2.sumabs()); }

inline RGBf operator+(const RGBf &v1, const RGBf &v2) {
    return RGBf( v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline RGBf operator+(const float v1, const RGBf &v2) {
	return RGBf( v1 + v2.e[0], v1 + v2.e[1], v1 + v2.e[2]);
}
inline RGBf operator+(const RGBf &v1, const float v2) {
	return RGBf( v1.e[0] + v2, v1.e[1] + v2, v1.e[2] + v2);
}

inline RGBf operator-(const RGBf &v1, const RGBf &v2) {
    return RGBf( v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline RGBf operator-(const float v1, const RGBf &v2) {
	return RGBf( v1 - v2.e[0], v1 - v2.e[1], v1 - v2.e[2]);
}

inline RGBf operator*(float t, const RGBf &v) {
    return RGBf(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

inline RGBf operator*(const RGBf &v, float t) {
    return RGBf(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

inline RGBf operator*(const RGBf &v1, const RGBf &v2) {
    return RGBf(v1.e[0]*v2.e[0], v1.e[1]*v2.e[1], v1.e[2]*v2.e[2]); 
}

inline RGBf operator/(const RGBf &v, float t) {
    return RGBf(v.e[0]/t, v.e[1]/t, v.e[2]/t); 
}

inline RGBf operator/(const RGBf &v1, const RGBf &v2) {
    return RGBf(v1.e[0]/v2.e[0], v1.e[1]/v2.e[1], v1.e[2]/v2.e[2]); 
}

inline RGBf& RGBf::operator+=(const RGBf &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

inline RGBf& RGBf::operator-=(const RGBf& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

inline RGBf& RGBf::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

inline RGBf& RGBf::operator/=(const float t) {
    e[0]  /= t;
    e[1]  /= t;
    e[2]  /= t;
    return *this;
}

inline RGBf& RGBf::operator/=(const RGBf& v) {
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

class RGB4f {
public:

	RGB4f() { }
	RGB4f(float r, float g, float b, float a = 0.0f) { data4 = _mm_set_ps(a,b,g,r); /* data[0] = r; data[1] = g; data[2] = b; */}
	RGB4f(const RGB4f &c) { data4 = c.data4; }
	//RGB4f(const RGB4f3f &c) { data[0] = c.r(); data[1] = c.g(); data[2] = c.b(); }
	RGB4f(const __m128 &c) { data4 = c; }

	RGB4f& operator=(const RGB4f& v) { data4 = v.data4; return *this; }
	RGB4f& operator=(const __m128& v) { data4 = v; return *this; }	
	float r() const { return data[0]; }
	float g() const { return data[1]; }
	float b() const { return data[2]; }
	float a() const { return data[3]; }
	float sum() const { return data[0] + data[1] + data[2]; }
	float sumabs() const { return fabs(data[0]) + fabs(data[1]) + fabs(data[2]); }
	void reset() { data4 = _mm_setzero_ps(); }

	RGB4f(const RGBf &vec)
	{
		data4 = _mm_set_ps(0.0f,vec.e[2],vec.e[1],vec.e[0]);
	}

	void setR(float r) {e[0] = r;}
	void setG(float g) {e[1] = g;}
	void setB(float b) {e[2] = b;}
	void setA(float a) {e[3] = a;}
	void set(const RGBf &c, float a = 0.0f) {data4 = _mm_set_ps(a,c.e[2],c.e[1],c.e[0]);}
	void set(const RGB4f &c) {data4 = _mm_set_ps(c.e[3],c.e[2],c.e[1],c.e[0]);}

	//RGB4f &operator=(const RGB4f3f &c) {
	//	data[0] = c.r(); data[1] = c.g(); data[2] = c.b();
	//	return *this;
	//}

	RGB4f operator+() const { return RGB4f( data[0], data[1], data[2]); }
	RGB4f operator-() const { return RGB4f(-data[0],-data[1],-data[2]); }
	float operator[](int i) const {assert(i >= 0 && i < 3); return data[i];}
	float& operator[](int i) {assert(i >= 0 && i < 3); return data[i];} 

	// RGB4f OPERATOR RGB4f
	RGB4f& operator+=(const RGB4f &c) { data4 = _mm_add_ps(data4, c.data4); return *this; } 
	RGB4f& operator-=(const RGB4f &c) { data4 = _mm_sub_ps(data4, c.data4); return *this;  } 
	RGB4f& operator*=(const RGB4f &c) { data4 = _mm_mul_ps(data4, c.data4); return *this;  } 
	RGB4f& operator/=(const RGB4f &c) { data4 = _mm_div_ps(data4, c.data4); return *this;  } 
	// RGB4f OPERATOR float
	RGB4f& operator+=(float f) { data4 = _mm_add_ps(data4, _mm_set1_ps(f)); return *this;  } 
	RGB4f& operator-=(float f) { data4 = _mm_sub_ps(data4, _mm_set1_ps(f)); return *this;  } 
	RGB4f& operator*=(float f) { data4 = _mm_mul_ps(data4, _mm_set1_ps(f)); return *this;  } 
	RGB4f& operator/=(float f) { data4 = _mm_div_ps(data4, _mm_set1_ps(f)); return *this; } 
	// RGB4f OPERATOR __m128
	RGB4f& operator+=(const __m128 &c) { data4 = _mm_add_ps(data4, c); return *this; } 
	RGB4f& operator-=(const __m128 &c) { data4 = _mm_sub_ps(data4, c); return *this;  } 
	RGB4f& operator*=(const __m128 &c) { data4 = _mm_mul_ps(data4, c); return *this;  } 
	RGB4f& operator/=(const __m128 &c) { data4 = _mm_div_ps(data4, c); return *this;  } 


	// RGB data as float values. The fourth value is for padding only
	// to make the structure 16-bytes long and therefore usable for SIMD
	// operations. The last index in the array is never used.
	__declspec(align(16)) union {
		float data[4];
		__m128 data4;
		float e[4];
	};
};


inline bool operator==(const RGB4f &c1, const RGB4f &c2) {
	return _mm_movemask_ps(_mm_cmpeq_ps(c1.data4, c2.data4)) == 7; 
}
inline bool operator!=(const RGB4f &c1, const RGB4f &c2) {
	return _mm_movemask_ps(_mm_cmpeq_ps(c1.data4, c2.data4)) != 7; 
}

inline bool operator<(const RGB4f &c1, const RGB4f &c2) {
	return (c1.sumabs() < c2.sumabs()); }
inline bool operator<=(const RGB4f &c1, const RGB4f &c2) {
	return (c1.sumabs() <= c2.sumabs()); }
inline bool operator>(const RGB4f &c1, const RGB4f &c2) {
	return (c1.sumabs() > c2.sumabs()); }
inline bool operator>=(const RGB4f &c1, const RGB4f &c2) {
	return (c1.sumabs() >= c2.sumabs()); }

inline RGB4f operator+(const RGB4f &c1, const RGB4f &c2) {
	return RGB4f(_mm_add_ps(c1.data4, c2.data4));}
inline RGB4f operator-(const RGB4f &c1, const RGB4f &c2) {
	return RGB4f(_mm_sub_ps(c1.data4, c2.data4));}
inline RGB4f operator*(const RGB4f &c1, const RGB4f &c2) {
	return RGB4f(_mm_mul_ps(c1.data4, c2.data4));}
inline RGB4f operator/(const RGB4f &c1, const RGB4f &c2) {
	return RGB4f(_mm_div_ps(c1.data4, c2.data4));}

inline RGB4f operator*(const RGB4f &c, float f) {
	return RGB4f(_mm_mul_ps(c.data4, _mm_set1_ps(f)));}
inline RGB4f operator*(float f, const RGB4f &c) {
	return RGB4f(_mm_mul_ps(c.data4, _mm_set1_ps(f)));}
inline RGB4f operator/(const RGB4f &c, float f) {
	return RGB4f(_mm_mul_ps(c.data4, _mm_set1_ps(1.0f/f)));}

inline RGB4f operator+(const RGB4f &c1, const __m128 &c2) {
	return RGB4f(_mm_add_ps(c1.data4, c2));}
inline RGB4f operator-(const RGB4f &c1, const __m128 &c2) {
	return RGB4f(_mm_sub_ps(c1.data4, c2));}
inline RGB4f operator*(const RGB4f &c1, const __m128 &c2) {
	return RGB4f(_mm_mul_ps(c1.data4, c2));}
inline RGB4f operator/(const RGB4f &c1, const __m128 &c2) {
	return RGB4f(_mm_div_ps(c1.data4, c2));}

inline RGB4f operator+(const __m128 &c1, const RGB4f &c2) {
	return RGB4f(_mm_add_ps(c1, c2.data4));}
inline RGB4f operator-(const __m128 &c1, const RGB4f &c2) {
	return RGB4f(_mm_sub_ps(c1, c2.data4));}
inline RGB4f operator*(const __m128 &c1, const RGB4f &c2) {
	return RGB4f(_mm_mul_ps(c1, c2.data4));}
inline RGB4f operator/(const __m128 &c1, const RGB4f &c2) {
	return RGB4f(_mm_div_ps(c1, c2.data4));}

inline std::ostream &operator<<(std::ostream &os, const RGB4f &t) {
	os << t[0] << " " << t[1] << " " << t[2];
	return os;
}

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

class RGBi
{
public:
	RGBi(unsigned int rgb = 0): data(rgb) {}
	RGBi(const RGBf &rgb)
	{
		set((unsigned int)(rgb.r()*255.0f), (unsigned int)(rgb.g()*255.0f), (unsigned int)(rgb.b()*255.0f));
	}

	unsigned int data;

	unsigned int r() {return data & 0xFF;}
	unsigned int g() {return (data >> 8) & 0xFF;}
	unsigned int b() {return (data >> 16) & 0xFF;}

	void setR(unsigned int r) {data |= (r & 0xFF);}
	void setG(unsigned int g) {data |= ((g & 0xFF) << 8);}
	void setB(unsigned int b) {data |= ((b & 0xFF) << 16);}

	void set(unsigned int r, unsigned int g, unsigned int b) {data = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16);} 

	static RGBi makeRGB(unsigned int r, unsigned int g, unsigned int b) 
	{
		RGBi temp;
		temp.data = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16);
		return temp;
	} 
};

class RGBc
{
public:
	unsigned char data[3];

	unsigned int r() {return data[0];}
	unsigned int g() {return data[1];}
	unsigned int b() {return data[2];}

	void setR(unsigned int r) {data[0] = (unsigned char)r;}
	void setG(unsigned int g) {data[1] = (unsigned char)g;}
	void setB(unsigned int b) {data[2] = (unsigned char)b;}
};

class HSVf
{
public:
	float h, s, v;

	HSVf(float h = 0.0f, float s = 0.0f, float v = 0.0f) : h(h), s(s), v(v) {}
	HSVf(const RGBf &rgb) {*this = RGBToHSV(rgb);}

	RGBf toRGB() {return HSVToRGB(*this);}

	static RGBf HSVToRGB(const HSVf &hsv)
	{
		// ######################################################################
		// T. Nathan Mundhenk
		// mundhenk@usc.edu
		// C/C++ Macro HSV to RGB

		float H = hsv.h;
		float S = hsv.s;
		float V = hsv.v;
		while (H < 0.0f) { H += 360.0f; };
		while (H >= 360.0f) { H -= 360.0f; };
		float R, G, B;
		if (V <= 0)
		{ R = G = B = 0; }
		else if (S <= 0)
		{
			R = G = B = V;
		}
		else
		{
			float hf = H / 60.0f;
			int i = (int)hf;
			float f = hf - i;
			float pv = V * (1 - S);
			float qv = V * (1 - S * f);
			float tv = V * (1 - S * (1 - f));
			switch (i)
			{

				// Red is the dominant color

			case 0:
				R = V;
				G = tv;
				B = pv;
				break;

				// Green is the dominant color

			case 1:
				R = qv;
				G = V;
				B = pv;
				break;
			case 2:
				R = pv;
				G = V;
				B = tv;
				break;

				// Blue is the dominant color

			case 3:
				R = pv;
				G = qv;
				B = V;
				break;
			case 4:
				R = tv;
				G = pv;
				B = V;
				break;

				// Red is the dominant color

			case 5:
				R = V;
				G = pv;
				B = qv;
				break;

				// Just in case we overshoot on our math by a little, we put these here. Since its a switch it won't slow us down at all to put these here.

			case 6:
				R = V;
				G = tv;
				B = pv;
				break;
			case -1:
				R = V;
				G = pv;
				B = qv;
				break;

				// The color is not defined, we should throw an error.

			default:
				//LFATAL("i Value error in Pixel conversion, Value is %d", i);
				R = G = B = V; // Just pretend its black/white
				break;
			}
		}

		return RGBf(R, G, B);
	}

	static HSVf RGBToHSV (const RGBf &rgb) {
#		ifndef min
#		define min(a, b) ((a) < (b) ? (a) : (b))
#		endif
#		ifndef max
#		define max(a, b) ((a) > (b) ? (a) : (b))
#		endif

		float computedH = 0.0f;
		float computedS = 0.0f;
		float computedV = 0.0f;

		//remove spaces from input RGB values, convert to int
		float r = rgb.r();
		float g = rgb.g();
		float b = rgb.b();

		float minRGB = min(r, min(g,b));
		float maxRGB = max(r, max(g,b));

		// Black-gray-white
		if (minRGB == maxRGB) {
			computedV = minRGB;
			return HSVf(0.0f, 0.0f, computedV);
		}

		// Colors other than black-gray-white:
		float d = (r==minRGB) ? g-b : ((b==minRGB) ? r-g : b-r);
		float h = (r==minRGB) ? 3.0f : ((b==minRGB) ? 1.0f : 5.0f);
		computedH = 60.0f*(h - d/(maxRGB - minRGB));
		computedS = (maxRGB - minRGB)/maxRGB;
		computedV = maxRGB;
		return HSVf(computedH, computedS, computedV);
	}


};

#endif
