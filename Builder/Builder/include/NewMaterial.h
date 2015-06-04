#ifndef NEW_MATERIAL_H
#define NEW_MATERIAL_H

#include "Vector3.h"

class NewMaterial
{
public:
	NewMaterial()
	{ 
		setDefault();
	};

	~NewMaterial() {};

	void clear()
	{
		mat_Ka = Vector3(0.0f, 0.0f, 0.0f);
		mat_Kd = Vector3(0.0f, 0.0f, 0.0f);
		mat_Ks = Vector3(0.0f, 0.0f, 0.0f);
		mat_Tf = Vector3(0.0f, 0.0f, 0.0f);
		mat_d = 0.f;
		mat_Ns = 0.f;
		mat_illum = 2; //default

		name[0] = 0;
	}

	void setDefault()
	{
		clear();
		mat_Kd = Vector3(1.0f, 1.0f, 1.0f);	// default NewMaterial
		mat_Tf = Vector3(1.0f, 1.0f, 1.0f);
		mat_d = 1.0f;
	}

	bool hasDiffuse() {return mat_Kd > Vector3(0,0,0);}
	bool hasSpecular() {return mat_Ks > Vector3(0,0,0);}
	
	void setMatKa(const Vector3 &newKa) {mat_Ka = newKa;}
	void setMatKd(const Vector3 &newKd) {mat_Kd = newKd;}
	void setMatKs(Vector3 &newKs) {mat_Ks = newKs;}
	void setMatTf(Vector3 &newTf) {mat_Tf = newTf;}
	void setMat_d(float &newMat_d) {mat_d = newMat_d;}
	void setMat_Ns(float &newMat_Ns) {mat_Ns = newMat_Ns;}
	void setMat_illum(int &illum) {mat_illum = illum;}
	void setName( const char *matName ) {strcpy_s(name, 59, matName);}

	const Vector3 &getMatKa() const {return mat_Ka;}
	const Vector3 &getMatKd() const {return mat_Kd;}
	const Vector3 &getMatKs() const {return mat_Ks;}
	const Vector3 &getMatTf() const {return mat_Tf;}
	float getMat_d() const {return mat_d;}
	float getMat_Ns() const {return mat_Ns;}
	int getMat_illum() const {return mat_illum;}
	const char *getName() const {return name;}

	bool isPerfectSpecular(float mat_Ns) const {return mat_Ns > 2047.0f;}
	bool isPerfectSpecular() const {return mat_Ns > 2047.0f;}

protected:
	
	Vector3 mat_Ka;			// ambient reflectance
	Vector3 mat_Kd;			// diffuse	reflectance
	Vector3 mat_Ks;			// specular reflectance
	Vector3 mat_Tf;			// transmission filter
	float mat_d;			// dissolve, (1(default): opaque, 0: transparent)
	float mat_Ns;			// specular exponent
	int mat_illum;			// illumination model

	char name[60];
};

typedef std::vector<NewMaterial> NewMaterialList;

#endif