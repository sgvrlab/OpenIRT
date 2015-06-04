#include "defines.h"
#include "CommonHeaders.h"

#include "Material.h"
#include "TextureManager.h"

#include <io.h>

using namespace irt;

void Material::setMapKa(const char *mapFileName) 
{
	TextureManager *texMan = TextureManager::getSingletonPtr();
	map_Ka = texMan->loadTexture(mapFileName);
}

void Material::setMapKd(const char *mapFileName) 
{
	TextureManager *texMan = TextureManager::getSingletonPtr();
	map_Kd = texMan->loadTexture(mapFileName);
}

void Material::setMapBump( const char *mapFileName ) 
{
	TextureManager *texMan = TextureManager::getSingletonPtr();
	map_bump = texMan->loadTexture(mapFileName);
}

std::string Trim(const std::string& s)
{

	size_t f,e ;

	if (s.length() == 0)
		return s;

	if (s.c_str()[0] == 10)
		return "";

	f = s.find_first_not_of(" \t\r\n");
	e = s.find_last_not_of(" \t\r\n");

	if (f == std::string::npos)
		return "";
	return std::string(s,f,e-f+1);
}

namespace irt
{
bool loadMaterialFromMTL(const char *fileName, MaterialList &matList)
{
	FILE *fp;
	errno_t err;

	char workingDirectory[MAX_PATH];
	strcpy_s(workingDirectory, MAX_PATH-1, fileName);
	// remove last entry of prompt
	for(int i=(int)strlen(workingDirectory)-1;i>=0;i--)
	{
		if(workingDirectory[i] == '/' || workingDirectory[i] == '\\')
		{
			workingDirectory[i] = 0;
			break;
		}
		workingDirectory[i] = 0;
	}


	if(err = fopen_s(&fp, fileName, "r")) return false;

	// parse MTL file
	char currentLine[500];
	Material mat;
	bool isFirstMat = true;
	while(fgets(currentLine, 499, fp))
	{
		if(strstr(currentLine, "newmtl"))
		{
			std::string curMatName = currentLine+7;
			curMatName = Trim(curMatName);

			if(!isFirstMat) 
			{
				matList.push_back(mat);
				mat.setDefault();
			}

			mat.setName(curMatName.c_str());

			isFirstMat = false;

			continue;
		}

		std::string curStr = currentLine;
		curStr = Trim(curStr);
		const char *curC = curStr.c_str();

		RGBf tmpReflectance;

		if(strstr(curC, "#")) continue;		// skip comments

		if(strstr(curC, "Ka") == curC)
		{
			sscanf_s(curC, "Ka %f %f %f", 
				&tmpReflectance.e[0], 
				&tmpReflectance.e[1], 
				&tmpReflectance.e[2]);
			mat.setMatKa(tmpReflectance);
		}
		if(strstr(curC, "Kd") == curC)
		{
			sscanf_s(curC, "Kd %f %f %f", 
				&tmpReflectance.e[0], 
				&tmpReflectance.e[1], 
				&tmpReflectance.e[2]);
			mat.setMatKd(tmpReflectance);
		}
		if(strstr(curC, "Ks") == curC)
		{
			sscanf_s(curC, "Ks %f %f %f", 
				&tmpReflectance.e[0], 
				&tmpReflectance.e[1], 
				&tmpReflectance.e[2]);
			mat.setMatKs(tmpReflectance);
		}
		if(strstr(curC, "Tf") == curC)
		{
			sscanf_s(curC, "Tf %f %f %f", 
				&tmpReflectance.e[0], 
				&tmpReflectance.e[1], 
				&tmpReflectance.e[2]);
			mat.setMatTf(tmpReflectance);
		}

		float tmpFloat;
		if(strstr(curC, "Ns") == curC)
		{
			sscanf_s(curC, "Ns %f", &tmpFloat);
			mat.setMat_Ns(tmpFloat);
		}
		if(strstr(curC, "d") == curC)
		{
			sscanf_s(curC, "d %f", &tmpFloat);
			mat.setMat_d(tmpFloat);
		}

		int tmpInt;
		if(strstr(curC, "illum") == curC)
		{
			sscanf_s(curC, "illum %d", &tmpInt);
			mat.setMat_illum(tmpInt);
		}

		if(strstr(curC, "map_Ka") == curC)
		{
			char mapFileName[MAX_PATH] = {0, };
			sscanf_s(curC, "map_Ka %s", mapFileName, MAX_PATH);
			if(strlen(mapFileName) != 0) 
			{
				char fullFileName[MAX_PATH];
				sprintf_s(fullFileName, MAX_PATH, "%s\\%s", workingDirectory, mapFileName);
				FILE *fp;
				fopen_s(&fp, fullFileName, "r");
				if(!fp) sprintf_s(fullFileName, MAX_PATH, "%s\\texture\\%s", workingDirectory, mapFileName);
				else fclose(fp);
				mat.setMapKa(fullFileName);
			}
		}

		if(strstr(curC, "map_Kd") == curC)
		{
			char mapFileName[MAX_PATH] = {0, };
			sscanf_s(curC, "map_Kd %s", mapFileName, MAX_PATH);
			if(strlen(mapFileName) != 0) 
			{
				char fullFileName[MAX_PATH];
				sprintf_s(fullFileName, MAX_PATH, "%s\\%s", workingDirectory, mapFileName);
				FILE *fp;
				fopen_s(&fp, fullFileName, "r");
				if(!fp) sprintf_s(fullFileName, MAX_PATH, "%s\\texture\\%s", workingDirectory, mapFileName);
				else fclose(fp);
				mat.setMapKd(fullFileName);
			}
		}

		if(strstr(curC, "map_bump") == curC)
		{
			char mapFileName[MAX_PATH] = {0, };
			sscanf_s(curC, "map_bump %s", mapFileName, MAX_PATH);
			if(strlen(mapFileName) != 0) 
			{
				char fullFileName[MAX_PATH];
				sprintf_s(fullFileName, MAX_PATH, "%s\\%s", workingDirectory, mapFileName);
				FILE *fp;
				fopen_s(&fp, fullFileName, "r");
				if(!fp) sprintf_s(fullFileName, MAX_PATH, "%s\\texture\\%s", workingDirectory, mapFileName);
				else fclose(fp);
				mat.setMapBump(fullFileName);
			}
		}
	}
	matList.push_back(mat);
	fclose(fp);
	return true;
}

bool saveMaterialToMTL(const char *fileName, MaterialList &matList)
{
	FILE *fp;
	errno_t err;

	if(err = fopen_s(&fp, fileName, "w")) return false;

	fprintf(fp, "#\n");
	fprintf(fp, "# Wavefront material file: %s\n", fileName);
	fprintf(fp, "#\n\n");

	for(int i=0;i<(int)matList.size();i++)
	{
		const Material &material = matList[i];
		fprintf(fp, "newmtl %s\n", material.getName());
		RGBf col;
		col = material.getMatKa();
		fprintf(fp, "\tKa %f %f %f\n", col.r(), col.g(), col.b());
		col = material.getMatKd();
		fprintf(fp, "\tKd %f %f %f\n", col.r(), col.g(), col.b());
		col = material.getMatKs();
		fprintf(fp, "\tKs %f %f %f\n", col.r(), col.g(), col.b());
		col = material.getMatTf();
		fprintf(fp, "\tTf %f %f %f\n", col.r(), col.g(), col.b());
		fprintf(fp, "\td %f\n", material.getMat_d());
		fprintf(fp, "\tNs %f\n", material.getMat_Ns());
		fprintf(fp, "\tillum %d\n", material.getMat_illum());

		char mapFileName[256];
		if(material.getMapKa())
		{
			strcpy_s(mapFileName, 256, strstr(material.getMapKa()->getFileName(), "texture")+8);
			fprintf(fp, "\tmap_Ka %s\n", mapFileName);
		}
		if(material.getMapKd())
		{
			strcpy_s(mapFileName, 256, strstr(material.getMapKd()->getFileName(), "texture")+8);
			fprintf(fp, "\tmap_Kd %s\n", mapFileName);
		}
		if(material.getMapBump())
		{
			strcpy_s(mapFileName, 256, strstr(material.getMapBump()->getFileName(), "texture")+8);
			fprintf(fp, "\tmap_bump %s\n", mapFileName);
		}
	}

	fclose(fp);

	return true;
}

bool generateMTLFromOOCMaterial(const char *oocFileName, const char *mtlFileName)
{
	typedef struct OOCMaterial_t
	{
		char dummy[52];
		Vector3 color;
	} OOCMaterial;

	FILE *fpSrc, *fpMTL;
	if(fopen_s(&fpSrc, oocFileName, "rb") != 0) return false;

	int fileSize = (int)_filelength(_fileno(fpSrc));

	if(fopen_s(&fpMTL, mtlFileName, "w") != 0)
	{
		fclose(fpSrc);
		return false;
	}

	int index = 0;
	OOCMaterial oocMaterial;
	while(fread_s(&oocMaterial, sizeof(OOCMaterial), sizeof(OOCMaterial), 1, fpSrc))
	{
		fprintf(fpMTL, "newmtl material%d\n", index++);
		fprintf(fpMTL, "\tKd %f %f %f\n", oocMaterial.color.x(), oocMaterial.color.y(), oocMaterial.color.z());
	}

	fclose(fpSrc);
	fclose(fpMTL);

	return index == fileSize/sizeof(OOCMaterial);
}

void modifyMaterial(MaterialList &matList, Material which, Material to)
{
	char name[256];
	for(int i=0;i<(int)matList.size();i++)
	{
		strcpy_s(name, 256, matList[i].getName());
		if(matList[i] == which)
		{
			matList[i] = to;
			matList[i].setName(name);
		}
	}
}
};