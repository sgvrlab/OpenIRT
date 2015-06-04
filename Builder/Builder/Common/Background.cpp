#include "stdafx.h"
#include "common.h"

void BackgroundConstant::shade(Vector3 &direction, rgb &backgroundColor) {
	backgroundColor = color;
}

void BackgroundEMSpherical::shade(Vector3 &direction, rgb &backgroundColor) {
	if (texPtr) {
		assert(fabs(direction.length() - 1.0f) < EPSILON);

		// calculate spherical coordinates
		float rho = acosf(-direction.y());
		float phi = direction.x() / sinf(rho);

		// check for numerical stability, otherwise acos() might
		// get an input > 1.0 which gives an undefined result..
		if (fabs(phi) <= 1.0f) {
			if (direction.z() >= 0.0f)
				phi = acosf(phi);
			else
				phi = 2*PI - acosf(phi);
		}
		else {
			phi = 0;
		}

		// map to [0..1)
		rho /= PI;
		phi /= 2*PI;

		texPtr->getTexValue(backgroundColor, phi, 1.0f - rho);


		
	}
}

void BackgroundEMCubeMap::shade(Vector3 &direction, rgb &backgroundColor) {
	// Order: Right, Left, Up, Down, Back, Front	
	static int faceToU[] = { 2, 2, 0, 0, 0, 0 };
	static int faceToV[] = { 1, 1, 2, 2, 1, 1 };
	static float faceToUSign[] = { 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f };
	static float faceToVSign[] = { 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f };

	int idx = direction.indexOfMaxAbsComponent();
	int face = (direction.e[idx] > 0.0)?0:1;
	face += 2*idx;

	//int idx1 = min((idx+1)%3, (idx+2)%3);
	//int idx2 = max((idx+1)%3, (idx+2)%3);
	int idx1 = faceToU[face];
	int idx2 = faceToV[face];

	float u = (faceToUSign[face]*direction.e[idx1] / fabs(direction.e[idx]) + 1.0f) / 2.0f;
	float v = (faceToVSign[face]*direction.e[idx2] / fabs(direction.e[idx]) + 1.0f) / 2.0f;

	texPtr[face]->getTexValue(backgroundColor, u, 1.0f - v);
}