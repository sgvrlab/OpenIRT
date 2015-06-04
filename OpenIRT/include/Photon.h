#pragma once

#include <vector>
#include "Vector3.h"
#include "RGB.h"

namespace irt
{

class Photon
{
public:
	enum SplitChoice
	{
		SPLIT_CHOICE_ROUND_ROBIN,
		SPLIT_CHOICE_HIGHEST_VARIANCE,
		SPLIT_CHOICE_LONGEST_DIM
	};

	enum SplitAxis
	{
		SPLIT_AXIS_X,
		SPLIT_AXIS_Y,
		SPLIT_AXIS_Z,
		SPLIT_AXIS_LEAF,
		SPLIT_AXIS_NULL
	};

	RGBf power;
	unsigned short axis;
	unsigned char theta, phi;
	Vector3 pos;
	float dummy;

	inline void setDirection(const Vector3 &dir)
	{
		theta = (unsigned char)(acosf(dir.z()) * 256.0f / PI);
		float temp = atan2f(dir.y(), dir.x());
		// change range: (-pi, pi] -> [0, 2pi)
		temp = temp < 0 ? -temp : temp;
		phi = (unsigned char)(temp * 256.0f / (2*PI));
	}
};

typedef struct PhotonVoxel_t
{
	RGBf power;
	float dummy;
} PhotonVoxel;


typedef std::vector<Photon> PhotonList;

};