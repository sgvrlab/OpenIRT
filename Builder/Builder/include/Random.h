#ifndef COMMON_RANDOM_H_
#define COMMON_RANDOM_H_


#include <windows.h>

/********************************************************************
	created:	2004/06/08
	created:	8.6.2004   0:37
	filename: 	c:\MSDev\MyProjects\Renderer\Common\Random.h
	file base:	Random
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Generates simple pseudo-random numbers using linear
				congruence with a seed.	
*********************************************************************/

class RandomLinear {
public:
	RandomLinear(unsigned long long createSeed = 0) {
		if (createSeed > 0)
			seed = createSeed;
		else
			seed = GetTickCount();
		mult = 62089911ULL;
		llong_max = 4294967295ULL;
		float_max = 4294967295.0f;
	}

	inline float sample();
	
	unsigned long long seed,		// seed (start-value)
					   mult,		// multiplicator
					   llong_max;	// max value for unsigned long long
	float			   float_max;	// max value for float

};

/**
* Generate a random number
*/
FORCEINLINE float RandomLinear::sample() {
	seed = mult * seed;
	return float(seed % llong_max) / float_max;
}

#endif // COMMON_RANDOM_H_