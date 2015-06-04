#ifndef COMMON_SAMPLE_H
#define COMMON_SAMPLE_H

/********************************************************************
	created:	2004/10/06
	created:	6.10.2004   17:56
	filename: 	c:\MSDev\MyProjects\Renderer\Common\Sample.h
	file base:	Sample
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Common functions for sampling in several dimensions
	            as well as filtering
*********************************************************************/

#include "Vector2.h"
#include "Random.h"

class Sampler {
public:

	Sampler() {
		generator = new RandomLinear();
	}

	~Sampler() {
		if (generator)
			delete generator;
	}

	/**
	* Generate num random numbers (0..1) and place 
	* them into array *dest.
	*/ 
	void random1(float *dest, int num);

	/**
	* Generate 2*num random numbers (0..1,0..1) and place 
	* them into array of vectors *dest.
	*/ 
	void random2(Vector2 *dest, int num);

	/**
	* Generate num random numbers (0..1) with jittering and place 
	* them into array *dest.
	*/ 
	void jitter1(float *dest, int num);

	/**
	* Generate 2*num^2 random numbers (0..1,0..1) with jittering and place 
	* them into array of vectors *dest.
	*/ 
	void jitter2(Vector2 *dest, int num);

	/**
	* Distribute samples according to box filter (do nothing except
	* shift by 0.5..)
	*/
	void filterBox(Vector2 *samples, int num) {
		for (int i = 0; i < num; i++) {
			samples[i].e[0] -= 0.5f;
			samples[i].e[1] -= 0.5f;			
		}
	}

	/**
	 * Distribute samples according to tent function
	 */
	void filterTent(Vector2 *samples, int num) {
		for (int i = 0; i < num; i++) {
			float x = samples[i].x();
			float y = samples[i].y();

			if (x < 0.5f)
				samples[i].setX((float)sqrt(2.0 * (double)x) - 1.0f);
			else
				samples[i].setX(1.0f - (float)sqrt(2.0 - 2.0 * (double)x));

			if (y < 0.5f)
				samples[i].setY((float)sqrt(2.0 * (double)y) - 1.0f);
			else
				samples[i].setY(1.0f - (float)sqrt(2.0 - 2.0 * (double)y));
		}
	}

	// TODO: different filters (cubic..)

protected:
	RandomLinear *generator;
};

#endif