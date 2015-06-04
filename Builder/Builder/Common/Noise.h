#ifndef COMMON_NOISE_H
#define COMMON_NOISE_H

#include <stdlib.h>
#include <math.h>

#ifndef _WIN32
#include <sys/time.h>
#include <unistd.h>
#endif

#include "Random.h"
#include "Vector3.h"

class PerlinNoise {
public:
	PerlinNoise();
	PerlinNoise(int seed);
	float noise(const Vector3&) const;
	Vector3 vectorNoise(const Vector3&) const;
	Vector3 vectorTurbulence( const Vector3&, int) const ;
	float turbulence(const Vector3&, int) const;
	float dturbulence(const Vector3&, int, float) const;
	float omega(float) const;
	Vector3 gamma(int, int, int) const;
	int intGamma(int, int) const;
	float knot(int, int, int, Vector3&) const;
	Vector3 vectorKnot(int, int, int, Vector3&) const;

	Vector3 grad[16];
	int phi[16];
};


inline float PerlinNoise::omega(float t) const {
	t = (t > 0.0)? t : -t;
	return (t < 1.0)?  (-6*t*t*t*t*t + 15*t*t*t*t -10*t*t*t + 1) : 0.0f;
}

inline Vector3 PerlinNoise::gamma(int i, int j, int k) const
{
	int idx;
	idx = phi[abs(k)%16];
	idx = phi[abs(j + idx) % 16];
	idx = phi[abs(i + idx) % 16];
	return grad[idx];
}

inline float PerlinNoise::knot(int i, int j, int k, Vector3& v) const {
	return ( omega(v.x()) * omega(v.y()) * omega(v.z()) * (dot(gamma(i,j,k),v)) );
}

inline Vector3 PerlinNoise::vectorKnot( int i, int j, int k, Vector3& v)
const {
	return ( omega(v.x()) * omega(v.y()) * omega(v.z()) * gamma(i,j,k) );
}

inline int PerlinNoise::intGamma(int i, int j) const {
	int idx;
	idx = phi[abs(j)%16];
	idx = phi[abs(i + idx) % 16];
	return idx;
}


#endif
