#pragma once

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

namespace irt
{

class Face {
public:
	unsigned char n;		// number of vertex indices
	int *verts;				// vertex indices
};

typedef struct FaceStr_t
{
	int n;
	std::string *f;
} FaceStr;

class Material
{
public:
	Material()
	{ 
		setDefault();
	};

	~Material() {clear();};

	void clear()
	{
		mat_Ka = Vector3(0.0f, 0.0f, 0.0f);
		mat_Kd = Vector3(0.0f, 0.0f, 0.0f);
		mat_Ks = Vector3(0.0f, 0.0f, 0.0f);
		mat_Tf = Vector3(0.0f, 0.0f, 0.0f);
		mat_d = 0.f;
		mat_Ns = 0.f;
		mat_illum = 2; //default

		//if(map_Ka) delete map_Ka;
		//if(map_Kd) 
		//	delete map_Kd;
		//if(map_bump) delete map_bump;

		name[0] = 0;

		map_Ka_name[0] = 0;
		map_Kd_name[0] = 0;
		map_bump_name[0] = 0;

		rangeKd = 0.f;
		rangeKs = 0.f;

		//map_Ka = 0;
		//map_Kd = 0;
		//map_bump = 0;
	}

	void setDefault()
	{
		//map_Ka = 0;
		//map_Kd = 0;
		//map_bump = 0;

		clear();
		mat_Kd = Vector3(1.0f, 1.0f, 1.0f);	// default material
		mat_Tf = Vector3(1.0f, 1.0f, 1.0f);
		mat_d = 1.0f;
	}

	bool hasDiffuse() {return mat_Kd > Vector3(0,0,0);}
	bool hasSpecular() {return mat_Ks > Vector3(0,0,0);}
	
	void setMatKa(const Vector3 &newKa) {mat_Ka = newKa;}
	void setMatKd(const Vector3 &newKd) {mat_Kd = newKd;}
	void setMatKs(const Vector3 &newKs) {mat_Ks = newKs;}
	void setMatTf(const Vector3 &newTf) {mat_Tf = newTf;}
	void setMat_d(const float newMat_d) {mat_d = newMat_d;}
	void setMat_Ns(const float newMat_Ns) {mat_Ns = newMat_Ns;}
	void setMat_illum(const int illum) {mat_illum = illum;}
	void setName( const char *matName ) {strcpy_s(name, 59, matName);}
	void setMapKa(const char *mapFileName) {strcpy_s(map_Ka_name, 255, mapFileName);}
	void setMapKd(const char *mapFileName) {strcpy_s(map_Kd_name, 255, mapFileName);}
	void setMapBump( const char *mapFileName ) {strcpy_s(map_bump_name, 255, mapFileName);}

	const Vector3 &getMatKa() const {return mat_Ka;}
	const Vector3 &getMatKd() const {return mat_Kd;}
	const Vector3 &getMatKs() const {return mat_Ks;}
	const Vector3 &getMatTf() const {return mat_Tf;}
	float getMat_d() const {return mat_d;}
	float getMat_Ns() const {return mat_Ns;}
	int getMat_illum() const {return mat_illum;}
	const char *getName() const {return name;}
	//BitmapTexture *getMapKa() const {return map_Ka;}
	//BitmapTexture *getMapKd() const {return map_Kd;}
	//BitmapTexture *getMapBump() const {return map_bump;}

	bool operator == (const Material &mat)
	{
		if(mat_Ka != mat.mat_Ka) return false;
		if(mat_Kd != mat.mat_Kd) return false;
		if(mat_Ks != mat.mat_Ks) return false;
		if(mat_Tf != mat.mat_Tf) return false;
		if(mat_d != mat.mat_d) return false;
		if(mat_Ns != mat.mat_Ns) return false;
		if(mat_illum != mat.mat_illum) return false;
		//if(map_Ka != mat.map_Ka) return false;
		//if(map_Kd != mat.map_Kd) return false;
		//if(map_bump != mat.map_bump) return false;
		return true;
	}

	bool operator != (const Material &mat)
	{
		return !(*this == mat);
	}

public:
	
	Vector3 mat_Ka;			// ambient reflectance
	Vector3 mat_Kd;			// diffuse	reflectance
	Vector3 mat_Ks;			// specular reflectance
	Vector3 mat_Tf;			// transmission filter
	float mat_d;			// dissolve, (1(default): opaque, 0: transparent)
	float mat_Ns;			// specular exponent
	int mat_illum;			// illumination model

	char name[60];

	float rangeKd;
	float rangeKs;

	char map_Ka_name[256];
	char map_Kd_name[256];
	char map_bump_name[256];

	//BitmapTexture *map_Ka;		// bitmap ambient
	//BitmapTexture *map_Kd;		// bitmap diffuse
	//BitmapTexture *map_bump;	// bitmap bump
};

typedef std::vector<Material> MaterialList;

class GroupInfo
{
public:
	std::string name;
	std::string materialFileName;
	std::string materialName;
};


};
