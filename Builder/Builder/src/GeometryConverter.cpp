#include "GeometryConverter.h"
#include <direct.h>
#include <io.h>
#include "FileMapper.h"
#include <hash_map>

GeometryConverter::TCReturnType GeometryConverter::convertTri(const char *fullFileName, bool useBackup)
{
	typedef struct OldTriangle_t {
		unsigned int p[3];		// vertex indices
		Vector3 n;			    // normal vector (normalized)
		float d;				// d from plane equation
		unsigned char  i1,i2;	// planes to be projected to
		unsigned short material;	// Index of material in list
	} OldTriangle;

	typedef struct NewTriangle_t {
		unsigned int p[3];		// vertex indices
		unsigned char  i1,i2;	// planes to be projected to
		unsigned short material;	// Index of material in list
		Vector3 n;			    // normal vector (normalized)
		float d;				// d from plane equation
	} NewTriangle;

	// extract file extension
	char fileName[256] = {0, }, fileExt[256] = {0, };
	for(int i=strlen(fullFileName)-1;i>=0;i--)
	{
		if(fullFileName[i] == '.')
		{
			strcpy_s(fileName, 255, fullFileName);
			fileName[i] = 0;
			strcpy_s(fileExt, 255, &fullFileName[i+1]);
			break;
		}
	}

	NewTriangle srcTri, dstTri;
	FILE *fpSrc, *fpDst;
	fopen_s(&fpSrc, fullFileName, "rb");

	// test triangle type
	fread_s(&srcTri, sizeof(NewTriangle), sizeof(NewTriangle), 1, fpSrc);
	if(srcTri.i1 >= 0 && srcTri.i1 <= 2 && srcTri.i2 >= 0 && srcTri.i2 <= 2)
	{
		//fclose(fpSrc);
		//return TYPE_NO_NEED;
	}

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

		fwrite(&dstTri, sizeof(NewTriangle), 1, fpDst);
	} while(fread_s(&oldTriangle, sizeof(OldTriangle), sizeof(OldTriangle), 1, fpSrc));
	
	fclose(fpSrc);
	fclose(fpDst);

	rename(fullFileName, oldFileName);
	rename(newFileName, fullFileName);

	if(!useBackup)
		unlink(oldFileName);

	return SUCCESS;
}

GeometryConverter::TCReturnType GeometryConverter::convertVert(const char *fullFileName, bool useBackup)
{
	typedef struct OldVertex_t {
		Vector3 v;				// vertex geometry
		Vector3 n;				// normal vector
		Vector3 c;				// color
		Vector2 uv;				// Texture coordinate
		unsigned char dummy[20];
	} OldVertex;

	typedef struct NewVertex_t {
		Vector3 v;				// vertex geometry
		float dummy1;
		Vector3 n;				// normal vector
		float dummy2;
		Vector3 c;				// color
		float dummy3;
		Vector2 uv;				// Texture coordinate
		unsigned char dummy[8];
	} NewVertex;

	// extract file extension
	char fileName[256] = {0, }, fileExt[256] = {0, };
	for(int i=strlen(fullFileName)-1;i>=0;i--)
	{
		if(fullFileName[i] == '.')
		{
			strcpy_s(fileName, 255, fullFileName);
			fileName[i] = 0;
			strcpy_s(fileExt, 255, &fullFileName[i+1]);
			break;
		}
	}

	NewVertex srcVert, dstVert;
	FILE *fpSrc, *fpDst;
	fopen_s(&fpSrc, fullFileName, "rb");

	// test vertex type
	fread_s(&srcVert, sizeof(NewVertex), sizeof(NewVertex), 1, fpSrc);
	if(srcVert.dummy1 == 0 && srcVert.dummy2 == 0 && srcVert.dummy3 == 0)
	{
		//fclose(fpSrc);
		//return TYPE_NO_NEED;
	}

	OldVertex oldVertex = *((OldVertex*)&srcVert);

	char oldFileName[256], newFileName[256];
	sprintf_s(newFileName, 255, "%s_new.%s", fileName, fileExt);
	sprintf_s(oldFileName, 255, "%s_old.%s", fileName, fileExt);

	fopen_s(&fpDst, newFileName, "wb");

	int a = sizeof(NewVertex);
	int b = sizeof(OldVertex);
	do
	{
		dstVert.v = oldVertex.v;
		dstVert.n = oldVertex.n;
		dstVert.c = oldVertex.c;
		dstVert.uv = oldVertex.uv;
		dstVert.dummy1 = dstVert.dummy2 = dstVert.dummy3 = 0;
		fwrite(&dstVert, sizeof(NewVertex), 1, fpDst);
	} while(fread_s(&oldVertex, sizeof(OldVertex), sizeof(OldVertex), 1, fpSrc));
	
	fclose(fpSrc);
	fclose(fpDst);

	rename(fullFileName, oldFileName);
	rename(newFileName, fullFileName);

	if(!useBackup)
		unlink(oldFileName);

	return SUCCESS;
}

GeometryConverter::TCReturnType GeometryConverter::convert(const char *filePath)
{
	char vertFileName[256];
	char triFileName[256];
	sprintf_s(vertFileName, "%s\\vertex.ooc", filePath);
	sprintf_s(triFileName, "%s\\tris.ooc", filePath);
	convertVert(vertFileName, false);
	convertTri(triFileName, false);
	return SUCCESS;
}

GeometryConverter::TCReturnType GeometryConverter::convertCluster(const char *filePath)
{
	char oldDir[256]; // Save working directory
	_getcwd(oldDir, 256);
	_chdir(filePath);		

	// Scan all files	
	_finddata_t found;
	long fileHandle;

	int numClusters = 0;
	if ((fileHandle = (long)_findfirst("tri_*.ooc", &found)) != -1) {
		do {
			numClusters++;
		} while(_findnext(fileHandle, &found) == 0);
		_findclose(fileHandle);
	}

	char singleVertFileName[256];
	char vertFileName[256];
	char triFileName[256];
	FILE *fpVert;
	FILE *fpTri;
	sprintf_s(singleVertFileName, "%s\\vertex.ooc", filePath);

	Vertex *verts = (Vertex*)FileMapper::map(singleVertFileName);

	for(int i=0;i<numClusters;i++)
	{
		sprintf_s(vertFileName, "%s\\vert_%d.ooc", filePath, i);
		sprintf_s(triFileName, "%s\\tri_%d.ooc", filePath, i);

		fopen_s(&fpTri, triFileName, "rb");

		int numTris = _filelength(fileno(fpTri))/sizeof(Triangle);

		Triangle *tris = new Triangle[numTris];
		fread_s(tris, sizeof(Triangle)*numTris, sizeof(Triangle), numTris, fpTri);

		fclose(fpTri);


		// localize vertices
		stdext::hash_map<unsigned int, unsigned int> mapG2L;
		typedef stdext::hash_map<unsigned int, unsigned int>::iterator MapIt;

		fopen_s(&fpVert, vertFileName, "wb");
		errno_t err = fopen_s(&fpTri, triFileName, "wb");

		/*
		Triangle tri;

		while(fread_s(&tri, sizeof(Triangle), sizeof(Triangle), 1, fpTri))
		{
			fwrite(&dstTri, sizeof(NewTriangle), 1, fpDst);
		} 
		*/
		for(int j=0;j<numTris;j++)
		{
			Triangle &tri = tris[j];

			for(int k=0;k<3;k++)
			{
				MapIt it = mapG2L.find(tri.p[k]);

				unsigned int newPos = -1;

				if(it == mapG2L.end())
				{
					fwrite(&verts[tri.p[k]], sizeof(Vertex), 1, fpVert);
					newPos = (unsigned int)mapG2L.size();
					mapG2L.insert(std::pair<unsigned int, unsigned int>(tri.p[k], newPos));
				}
				else
				{
					newPos = it->second;
				}

				tri.p[k] = newPos;
			}

			fwrite(&tri, sizeof(Triangle), 1, fpTri);
		}
	
		fclose(fpVert);
		fclose(fpTri);

		delete[] tris;

		convertVert(vertFileName, false);
		convertTri(triFileName, false);

		convert(filePath, NEW_OOC, SIMP, i);
	}

	FileMapper::unmap(verts);

	_chdir(oldDir);

	return SUCCESS;
}

GeometryConverter::TCReturnType GeometryConverter::convert(const char *filePath, GeomType typeFrom, GeomType typeTo, int cluster)
{
	typedef struct OldVertex_t {
		Vector3 v;				// vertex geometry
		Vector3 n;				// normal vector
		Vector3 c;				// color
		Vector2 uv;				// Texture coordinate
		unsigned char dummy[20];
	} OldVertex;

	typedef struct NewVertex_t {
		Vector3 v;				// vertex geometry
		float dummy1;
		Vector3 n;				// normal vector
		float dummy2;
		Vector3 c;				// color
		float dummy3;
		Vector2 uv;				// Texture coordinate
		unsigned char dummy[8];
	} NewVertex;

	typedef struct SimpVertex_t {
		Vector3 v;				// vertex geometry
	} SimpVertex;

	typedef struct OldTriangle_t {
		unsigned int p[3];		// vertex indices
		Vector3 n;			    // normal vector (normalized)
		float d;				// d from plane equation
		unsigned char  i1,i2;	// planes to be projected to
		unsigned short material;	// Index of material in list
	} OldTriangle;

	typedef struct NewTriangle_t {
		unsigned int p[3];		// vertex indices
		unsigned char  i1,i2;	// planes to be projected to
		unsigned short material;	// Index of material in list
		Vector3 n;			    // normal vector (normalized)
		float d;				// d from plane equation
	} NewTriangle;

	typedef struct SimpTriangle_t {
		unsigned int p[3];		// vertex indices
	} SimpTriangle;

	FILE *fpVert, *fpTri, *fpNode;
	FILE *fpVertConv, *fpTriConv;

	AABB bb;
	bool createBB = true;

	char vertFileName[256];
	char triFileName[256];
	char cVertFileName[256];
	char cTriFileName[256];
	char nodeFileName[256];
	sprintf_s(nodeFileName, "%s\\BB", filePath);

	if(cluster < 0)
	{
		sprintf_s(vertFileName, "%s\\vertex.ooc", filePath);
		sprintf_s(triFileName, "%s\\tris.ooc", filePath);
		sprintf_s(cVertFileName, "%s\\conv_vertex.ooc", filePath);
		sprintf_s(cTriFileName, "%s\\conv_tris.ooc", filePath);
	}
	else
	{
		sprintf_s(vertFileName, "%s\\vert_%d.ooc", filePath, cluster);
		sprintf_s(triFileName, "%s\\tri_%d.ooc", filePath, cluster);
		sprintf_s(cVertFileName, "%s\\conv_vert_%d.ooc", filePath, cluster);
		sprintf_s(cTriFileName, "%s\\conv_tri_%d.ooc", filePath, cluster);
	}

	fopen_s(&fpVert, vertFileName, "rb");
	fopen_s(&fpTri, triFileName, "rb");
	fopen_s(&fpVertConv, cVertFileName, "wb");
	fopen_s(&fpTriConv, cTriFileName, "wb");

	// open and read if BB exist
	fopen_s(&fpNode, nodeFileName, "rb");
	if(fpNode)
	{
		fread_s(&bb, sizeof(AABB), sizeof(AABB), 1, fpNode);
		fclose(fpNode);
	}
	fopen_s(&fpNode, nodeFileName, "wb");


	char dataSrc[1024];
	char dataDst[1024];
	size_t sizeOfFromVert = 0, sizeOfToVert = 0;
	size_t sizeOfFromTri = 0, sizeOfToTri = 0;

	Triangle tri;

	switch(typeFrom)
	{
	case OLD_OOC: sizeOfFromVert = sizeof(OldVertex); sizeOfFromTri = sizeof(OldTriangle); break;
	case NEW_OOC: sizeOfFromVert = sizeof(NewVertex); sizeOfFromTri = sizeof(NewTriangle); break;
	case SIMP: sizeOfFromVert = sizeof(SimpVertex); sizeOfFromTri = sizeof(SimpTriangle); break;
	}

	switch(typeTo)
	{
	case OLD_OOC: sizeOfToVert = sizeof(OldVertex); sizeOfToTri = sizeof(OldTriangle); break;
	case NEW_OOC: sizeOfToVert = sizeof(NewVertex); sizeOfToTri = sizeof(NewTriangle); break;
	case SIMP: sizeOfToVert = sizeof(SimpVertex); sizeOfToTri = sizeof(SimpTriangle); break;
	}

	while(fread_s(dataSrc, 1024, sizeOfFromVert, 1, fpVert))
	{
		Vertex vert;

		switch(typeFrom)
		{
		case OLD_OOC:
			{
				OldVertex *v = (OldVertex*)dataSrc;
				vert.v = v->v;
				vert.n = v->n;
				vert.c = v->c;
				vert.uv = v->uv;
			}
			break;
		case NEW_OOC:
			{
				NewVertex *v = (NewVertex*)dataSrc;
				vert.v = v->v;
				vert.n = v->n;
				vert.c = v->c;
				vert.uv = v->uv;
			}
			break;
		case SIMP:
			{
				SimpVertex *v = (SimpVertex*)dataSrc;
				vert.v = v->v;
				vert.n = Vector3(0.0f);
				vert.c = Vector3(0.0f);
				vert.uv = Vector2(0.0f, 0.0f);
			}
			break;
		}

		bb.update(vert.v);

		switch(typeTo)
		{
		case OLD_OOC:
			{
				OldVertex *v = (OldVertex*)dataDst;
				v->v = vert.v;
				v->n = vert.n;
				v->c = vert.c;
				v->uv = vert.uv;
			}
			break;
		case NEW_OOC:
			{
				NewVertex *v = (NewVertex*)dataDst;
				v->v = vert.v;
				v->n = vert.n;
				v->c = vert.c;
				v->uv = vert.uv;
			}
			break;
		case SIMP:
			{
				SimpVertex *v = (SimpVertex*)dataDst;
				v->v = vert.v;
			}
			break;
		}

		fwrite(dataDst, sizeOfToVert, 1, fpVertConv);
	}

	while(fread_s(dataSrc, sizeOfFromTri, sizeOfFromTri, 1, fpTri))
	{
		Triangle tri;

		switch(typeFrom)
		{
		case OLD_OOC:
			{
				OldTriangle *t = (OldTriangle*)dataSrc;
				tri.p[0] = t->p[0]; tri.p[1] = t->p[1]; tri.p[2] = t->p[2];
				tri.n = t->n;
				tri.d = t->d;
				tri.i1 = t->i1; tri.i2 = t->i2;
			}
			break;
		case NEW_OOC:
			{
				NewTriangle *t = (NewTriangle*)dataSrc;
				tri.p[0] = t->p[0]; tri.p[1] = t->p[1]; tri.p[2] = t->p[2];
				tri.n = t->n;
				tri.d = t->d;
				tri.i1 = t->i1; tri.i2 = t->i2;
			}
			break;
		case SIMP:
			{
				SimpTriangle *t = (SimpTriangle*)dataSrc;
				tri.p[0] = t->p[0]; tri.p[1] = t->p[1]; tri.p[2] = t->p[2];
				tri.n = Vector3(0.0f);
				tri.d = 0.0f;
				tri.i1 = 0; tri.i2 = 0;
			}
			break;
		}

		switch(typeTo)
		{
		case OLD_OOC:
			{
				OldTriangle *t = (OldTriangle*)dataDst;
				t->p[0] = tri.p[0]; t->p[1] = tri.p[1]; t->p[2] = tri.p[2];
				t->n = tri.n;
				t->d = tri.d;
				t->i1 = tri.i1; t->i2 = tri.i2;
			}
			break;
		case NEW_OOC:
			{
				NewTriangle *t = (NewTriangle*)dataDst;
				t->p[0] = tri.p[0]; t->p[1] = tri.p[1]; t->p[2] = tri.p[2];
				t->n = tri.n;
				t->d = tri.d;
				t->i1 = tri.i1; t->i2 = tri.i2;
			}
			break;
		case SIMP:
			{
				SimpTriangle *t = (SimpTriangle*)dataDst;
				t->p[0] = tri.p[0]; t->p[1] = tri.p[1]; t->p[2] = tri.p[2];
			}
			break;
		}

		fwrite(dataDst, sizeOfToTri, 1, fpTriConv);
	}

	fwrite(&bb, sizeof(AABB), 1, fpNode);

	fclose(fpVert);
	fclose(fpTri);
	fclose(fpNode);
	fclose(fpVertConv);
	fclose(fpTriConv);

	return SUCCESS;
}
