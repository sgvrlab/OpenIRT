#ifndef _BBOX_H_
#define _BBOX_H_

#include "common.h"

/************************************************************************/
/* ´ö¼ö                                                                 */
/************************************************************************/
#define MAX(a,b)	((a) > (b) ? (a) : (b))
#define MIN(a,b)	((a) < (b) ? (a) : (b))
#define BOX_MIN		pp[0]
#define BOX_MAX		pp[1]

inline void
vmin(Vector3 &a, const Vector3 &b)
{
	a.setX( MIN(a.e[0], b.e[0]) ) ;
	a.setY( MIN(a.e[1], b.e[1]) ) ;
	a.setZ( MIN(a.e[2], b.e[2]) ) ;
}

inline void
vmax(Vector3 &a, const Vector3 &b)
{
	a.setX( MAX(a.e[0], b.e[0]) ) ;
	a.setY( MAX(a.e[1], b.e[1]) ) ;
	a.setZ( MAX(a.e[2], b.e[2]) ) ;
}
/************************************************************************/

class BBox
{
public:
	BBox() {}
	BBox(const Vector3& a, const Vector3& b) { pp[0] = a; pp[1] = b; }
	Vector3 getmin() const { return pp[0]; }
	Vector3 getmax() const { return pp[1]; }
	bool rayIntersect(const Ray& r, float* result_min, float* result_max);
	void setValue(const Vector3* bb);
	Vector3 pp[2];

	/************************************************************************/
	/* ´ö¼ö                                                                 */
	/************************************************************************/
	FORCEINLINE BBox(const Vector3 &a, const Vector3 &b, int dummy ) {
		BOX_MIN = a;
		BOX_MAX = a;
		vmin(BOX_MIN, b);
		vmax(BOX_MAX, b);
	}

	FORCEINLINE void setValue(const Vector3 &a, const Vector3 &b ) {
		BOX_MIN = a ;
		BOX_MAX = a ;
		vmin(BOX_MIN, b);
		vmax(BOX_MAX, b);
	}

	FORCEINLINE bool overlaps(const BBox& b) const
	{
		if (BOX_MIN.e[0] > b.BOX_MAX.e[0]) return false;
		if (BOX_MIN.e[1] > b.BOX_MAX.e[1]) return false;
		if (BOX_MIN.e[2] > b.BOX_MAX.e[2]) return false;

		if (BOX_MAX.e[0] < b.BOX_MIN.e[0]) return false;
		if (BOX_MAX.e[1] < b.BOX_MIN.e[1]) return false;
		if (BOX_MAX.e[2] < b.BOX_MIN.e[2]) return false;

		return true;
	}

	FORCEINLINE BBox &operator += (const Vector3 &p)
	{
		vmin(BOX_MIN, p);
		vmax(BOX_MAX, p);
		return *this;
	}

	FORCEINLINE BBox &operator += (const BBox &b)
	{
		vmin(BOX_MIN, b.BOX_MIN);
		vmax(BOX_MAX, b.BOX_MAX);
		return *this;
	}

	FORCEINLINE BBox operator + ( const BBox &v) const
	{ BBox rt(*this); return rt += v; }
	/************************************************************************/
};

#endif