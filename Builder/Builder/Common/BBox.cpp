#include "stdafx.h"
#include "BBox.h"

bool BBox::rayIntersect(const Ray &r, float* result_min, float* result_max)
{
	float interval_min = -FLT_MAX;
	float interval_max = FLT_MAX;

	float t0 = (pp[r.posneg[3]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
	float t1 = (pp[r.posneg[0]].e[0] - r.data[0].e[0]) * r.data[2].e[0];
	if (t0 > interval_min) interval_min = t0;
	if (t1 < interval_max) interval_max = t1;
	if (interval_min > interval_max) return false;

	t0 = (pp[r.posneg[4]].e[1] - r.data[0].e[1]) * r.data[2].e[1];
	t1 = (pp[r.posneg[1]].e[1] - r.data[0].e[1]) * r.data[2].e[1];
	if (t0 > interval_min) interval_min = t0;
	if (t1 < interval_max) interval_max = t1;
	if (interval_min > interval_max) return false;

	t0 = (pp[r.posneg[5]].e[2] - r.data[0].e[2]) * r.data[2].e[2];
	t1 = (pp[r.posneg[2]].e[2] - r.data[0].e[2]) * r.data[2].e[2];
	if (t0 > interval_min) interval_min = t0;
	if (t1 < interval_max) interval_max = t1;

	*result_min = interval_min;
	*result_max = interval_max;
	return (interval_min <= interval_max);
}

void BBox::setValue(const Vector3* bb)
{
	pp[0] = bb[0];
	pp[1] = bb[1];
}
