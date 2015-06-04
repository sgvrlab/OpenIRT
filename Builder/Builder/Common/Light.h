#ifndef COMMON_LIGHT_H
#define COMMON_LIGHT_H

enum LightType {
	LIGHT_POINT,
	LIGHT_DIRECTIONAL,
	LIGHT_SPOT
};

/**
 * Structure for (non-area) lights.
 * 
 * Supports different light types: point, directional, spot.
 * Some members are optional and will only be relevant for a
 * certain type (e.g. pos is not necessary for directional lights)
 */
typedef struct Light_t {	
	Vector3 pos;			// Light position (if local)	
	Vector3 direction;		// Light direction (if not point)
	LightType type;			// Type of light (see above)
	Real intensity;			// Energy of light
	Real cutoff_distance;	// distance at which light intensity is zero
	rgb color;				// the light's color
} Light, *LightPtr;

typedef std::vector<Light> LightList;
typedef LightList::iterator LightListIterator;

/**
 * Light-emitting triangle.
 *
 * The emitter emits light in three wavelengths (r,g,b). The sum of
 * the three intensities is saved in summedIntensity for weighting this
 * emitter. Currently, diffuse emission is assumed, that is, the lights
 * direction is importance sampled according to the cosine function.
 * 
 */
typedef struct Emitter_t {	
	rgb emissionIntensity;		// emission separated into the three color bands
	rgb emissionColor;			// emission separated into the three color bands, but limited to 0.0-1.0
	float summedIntensity;		// sum of all three emission components, used for choosing emitters
	Vector3 p[3];				// the three vertices of the triangle
	Vector3 n;					// the normal
	float area, areaInv;		// area of this emitter

	// Samples a point on the light source given by the two random
	// variable u and v (in [0..1])
	void sample(const float u, const float v, Vector3 &pointOnEmitter) {
		float temp = sqrt(1.0f - u);
		float beta = 1.0f - temp;
		float gamma = temp*v;

		pointOnEmitter = (1.0f - beta - gamma)*p[0] + beta*p[1] + gamma*p[2];
	}

	// Samples a direction from this light source given by the two
	// random variables u and v (in [0..1])
	void sampleDirection(const float u, const float v, Vector3 &newDirection) {
		static Vector3 m1(1.0f, 0.0f, 0.0f);
		static Vector3 m2(0.0f, 1.0f, 0.0f);
		
		// Importance sample according to cosine
		float phi = 2.0f * PI * u;
		float r = sqrtf(v);
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0f - x*x - y*y);

		// build orthonormal basis from normal
		Vector3 U = cross(n, m1);
		if (U.length() < 0.01f)
			U = cross(n, m2); 
		Vector3 V = cross(n, U);

		// use coordinates in basis:
		newDirection = x * U + y * V + z * n;
	}

} Emitter, *EmitterPtr;

typedef std::vector<Emitter> EmitterList;
typedef EmitterList::iterator EmitterListIterator;

#endif