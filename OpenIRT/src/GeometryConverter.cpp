#include "GeometryConverter.h"

using namespace irt;

GeometryConverter::TCReturnType GeometryConverter::convertTri(const char *fullFileName)
{
	// extract file extension
	char fileName[256] = {0, }, fileExt[256] = {0, };
	for(int i=(int)strlen(fullFileName)-1;i>=0;i--)
	{
		if(fullFileName[i] == '.')
		{
			strcpy_s(fileName, 255, fullFileName);
			fileName[i] = 0;
			strcpy_s(fileExt, 255, &fullFileName[i+1]);
			break;
		}
	}

	Triangle srcTri, dstTri;
	FILE *fpSrc, *fpDst;
	fopen_s(&fpSrc, fullFileName, "rb");

	// test triangle type
	fread_s(&srcTri, sizeof(Triangle), sizeof(Triangle), 1, fpSrc);
	if(srcTri.i1 >= 0 && srcTri.i1 <= 2 && srcTri.i2 >= 0 && srcTri.i2 <= 2)
	{
		fclose(fpSrc);
		return TYPE_NO_NEED;
	}

	typedef struct OldTriangle_t {
		unsigned int p[3];		// vertex indices
		Vector3 n;			    // normal vector (normalized)
		float d;				// d from plane equation
		unsigned char  i1,i2;	// planes to be projected to
		unsigned short material;	// Index of material in list
	} OldTriangle;

	OldTriangle oldTriangle = *((OldTriangle*)&srcTri);

	char oldFileName[256], newFileName[256];
	sprintf_s(newFileName, 255, "%s_new.%s", fileName, fileExt);
	sprintf_s(oldFileName, 255, "%s_old.%s", fileName, fileExt);

	fopen_s(&fpDst, newFileName, "wb");

	do
	{
		dstTri.p[0] = oldTriangle.p[0];
		dstTri.p[1] = oldTriangle.p[1];
		dstTri.p[2] = oldTriangle.p[2];
		dstTri.i1 = oldTriangle.i1;
		dstTri.i2 = oldTriangle.i2;
		dstTri.material = oldTriangle.material;
		dstTri.n = oldTriangle.n;
		dstTri.d = oldTriangle.d;

		fwrite(&dstTri, sizeof(Triangle), 1, fpDst);
	} while(fread_s(&oldTriangle, sizeof(OldTriangle), sizeof(OldTriangle), 1, fpSrc));
	
	fclose(fpSrc);
	fclose(fpDst);

	rename(fullFileName, oldFileName);
	rename(newFileName, fullFileName);

	return SUCCESS;
}

GeometryConverter::TCReturnType GeometryConverter::convertVert(const char *fullFileName)
{
	// extract file extension
	char fileName[256] = {0, }, fileExt[256] = {0, };
	for(int i=(int)strlen(fullFileName)-1;i>=0;i--)
	{
		if(fullFileName[i] == '.')
		{
			strcpy_s(fileName, 255, fullFileName);
			fileName[i] = 0;
			strcpy_s(fileExt, 255, &fullFileName[i+1]);
			break;
		}
	}

	Vertex srcVert, dstVert;
	FILE *fpSrc, *fpDst;
	fopen_s(&fpSrc, fullFileName, "rb");

	// test vertex type
	fread_s(&srcVert, sizeof(Vertex), sizeof(Vertex), 1, fpSrc);
	if(srcVert.dummy1 == 0 && srcVert.dummy2 == 0 && srcVert.dummy3 == 0)
	{
		fclose(fpSrc);
		return TYPE_NO_NEED;
	}

	typedef struct OldVertex_t {
		Vector3 v;				// vertex geometry
		Vector3 n;				// normal vector
		Vector3 c;				// color
		Vector2 uv;				// Texture coordinate
		unsigned char dummy[20];
	} OldVertex;

	OldVertex oldVertex = *((OldVertex*)&srcVert);

	char oldFileName[256], newFileName[256];
	sprintf_s(newFileName, 255, "%s_new.%s", fileName, fileExt);
	sprintf_s(oldFileName, 255, "%s_old.%s", fileName, fileExt);

	fopen_s(&fpDst, newFileName, "wb");

	int a = sizeof(Vertex);
	int b = sizeof(OldVertex);
	do
	{
		dstVert.v = oldVertex.v;
		dstVert.n = oldVertex.n;
		dstVert.c = oldVertex.c;
		dstVert.uv = oldVertex.uv;
		dstVert.dummy1 = dstVert.dummy2 = dstVert.dummy3 = 0;
		fwrite(&dstVert, sizeof(Vertex), 1, fpDst);
	} while(fread_s(&oldVertex, sizeof(Vertex), sizeof(Vertex), 1, fpSrc));
	
	fclose(fpSrc);
	fclose(fpDst);

	rename(fullFileName, oldFileName);
	rename(newFileName, fullFileName);

	return SUCCESS;
}