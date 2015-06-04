#ifndef COMMON_RAY_H
#define COMMON_RAY_H

using std::ostream;
using std::istream;

class Ray  {
public:

	Ray() {}
	Ray(const Vector3& a, const Vector3& b) 
	{ 
		data[0] = a; data[1] = b; 
		data[2] = Vector3(1.0f / b.x(), 1.0f / b.y(), 1.0f / b.z());

		posneg[0] =  (data[1].x() >= 0 ? 0 : 1);
		posneg[1] = posneg[0] ^ 1;

		posneg[2] =  (data[1].y() >= 0 ? 0 : 1);
		posneg[3] = posneg[2] ^ 1;  

		posneg[4] =  (data[1].z() >= 0 ? 0 : 1);
		posneg[5] = posneg[4] ^ 1;  
	}

	Ray(const Ray& r) {*this = r;}
	Vector3 origin() const {return data[0];}
	Vector3 direction() const {return data[1];}
	Vector3 invDirection() const {return data[2];}
	void setOrigin(const Vector3& v) {data[0] = v;}
	void setOrigin(const float *v) {data[0].set(v);}
	void setDirection(const Vector3& v) 
	{
		data[1] = v;
		data[2] = Vector3(1.0f / v.x(), 1.0f / v.y(), 1.0f / v.z());

		posneg[0] =  (data[1].x() >= 0 ? 0 : 1);
		posneg[1] = posneg[0] ^ 1;

		posneg[2] =  (data[1].y() >= 0 ? 0 : 1);
		posneg[3] = posneg[2] ^ 1;  

		posneg[4] =  (data[1].z() >= 0 ? 0 : 1);
		posneg[5] = posneg[4] ^ 1;  
	}
	void setDirection(const float *v) 
	{
		data[1].set(v);
		data[2] = Vector3(1.0f / v[0], 1.0f / v[1], 1.0f / v[2]);

		posneg[0] =  (data[1].x() >= 0 ? 0 : 1);
		posneg[1] = posneg[0] ^ 1;

		posneg[2] =  (data[1].y() >= 0 ? 0 : 1);
		posneg[3] = posneg[2] ^ 1;  

		posneg[4] =  (data[1].z() >= 0 ? 0 : 1);
		posneg[5] = posneg[4] ^ 1;  
	}

	Vector3 pointAtParameter(float t) const { return data[0] + t*data[1]; }

	// Index 0: origin
	// Index 1: direction
	// Index 2: inverse of direction
	Vector3 data[3];
	// Direction sign and negated sign of each of the direction's components
	int posneg[6]; 
};

inline std::ostream &operator<<(std::ostream &os, const Ray &r) {
	os << "(" << r.origin() << ") + t("  << r.direction() << ")";
	return os;
}


#endif
