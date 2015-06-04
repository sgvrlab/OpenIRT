#include "stdafx.h"
#include "Sample.h"

/**
* Generate num random numbers (0..1) and place 
* them into array *dest.
*/ 
void Sampler::random1(float *dest, int num) {
	for (int i=0; i<num; i++) {
		dest[i] = generator->sample();
	}
}

/**
* Generate 2*num random numbers (0..1,0..1) and place 
* them into array of vectors *dest.
*/ 
void Sampler::random2(Vector2 *dest, int num) {
	for (int i=0; i<num; i++) {
		dest[i].e[0] = generator->sample();
		dest[i].e[1] = generator->sample();
	}
}

/**
* Generate num random numbers (0..1) with jittering and place 
* them into array *dest.
*/ 
void Sampler::jitter1(float *dest, int num) {
	for (int i=0; i<num; i++) {
		dest[i] = ((float)i + generator->sample()) / (float)num;
	}
}

/**
* Generate num^2 random numbers (0..1,0..1) with jittering and place 
* them into array of vectors *dest.
*/ 
void Sampler::jitter2(Vector2 *dest, int num) {
	float numInv = 1.0f / (float)num;
	for (int i=0; i<num; i++) {
		for (int j=0; j<num; j++) {
			float x = ((float)i + generator->sample()) * numInv;
			float y = ((float)j + generator->sample()) * numInv;
			dest[i*num + j].e[0] = x;
			dest[i*num + j].e[1] = y;
		}
	}
}

