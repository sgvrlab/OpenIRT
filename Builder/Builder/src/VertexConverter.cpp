#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "OptionManager.h"
#include "BVHNodeDefine.h"
#include "Vertex.h"
#include <io.h>

#include "VertexConverter.h"

int VertexConverter::Do(const char* filepath, const char* vtype)
{
	char fileNameIn[MAX_PATH], fileNameOut[MAX_PATH];
	sprintf(fileNameIn, "%s/vertex.ooc", filepath);
	sprintf(fileNameOut, "%s/vertex_%s.ooc", filepath, vtype);
	FILE *fpIn, *fpOut;
	fpIn = fopen(fileNameIn, "rb");
	fpOut = fopen(fileNameOut, "wb");

	printf("%s\n", fileNameIn);

	VertexType type;
	if(strcmp(vtype, "V") == 0) type = V;
	if(strcmp(vtype, "VN") == 0) type = VN;
	if(strcmp(vtype, "VC") == 0) type = VC;
	if(strcmp(vtype, "VT") == 0) type = VT;
	if(strcmp(vtype, "VNC") == 0) type = VNC;
	if(strcmp(vtype, "VCT") == 0) type = VCT;
	if(strcmp(vtype, "VNT") == 0) type = VNT;
	if(strcmp(vtype, "VNCT") == 0) type = VNCT;

	Vertex vertIn;
	while(0 != fread(&vertIn, sizeof(Vertex), 1, fpIn))
	{
		switch(type)
		{
		case V :
			{
				VertexV vertOut;
				vertOut.v = vertIn.v;
				fwrite(&vertOut, sizeof(VertexV), 1, fpOut);
			}
			break;
		case VN :
			{
				VertexVN vertOut;
				vertOut.v = vertIn.v;
				vertOut.n = vertIn.n;
				fwrite(&vertOut, sizeof(VertexVN), 1, fpOut);
			}
			break;
		case VC :
			{
				VertexVC vertOut;
				vertOut.v = vertIn.v;
				vertOut.c = vertIn.c;
				fwrite(&vertOut, sizeof(VertexVC), 1, fpOut);
			}
			break;
		case VT :
			{
				VertexVT vertOut;
				vertOut.v = vertIn.v;
				vertOut.uv = vertIn.uv;
				fwrite(&vertOut, sizeof(VertexVT), 1, fpOut);
			}
			break;
		case VNC :
			{
				VertexVNC vertOut;
				vertOut.v = vertIn.v;
				vertOut.n = vertIn.n;
				vertOut.c = vertIn.c;
				fwrite(&vertOut, sizeof(VertexVNC), 1, fpOut);
			}
			break;
		case VCT :
			{
				VertexVCT vertOut;
				vertOut.v = vertIn.v;
				vertOut.c = vertIn.c;
				vertOut.uv = vertIn.uv;
				fwrite(&vertOut, sizeof(VertexVCT), 1, fpOut);
			}
			break;
		case VNT :
			{
				VertexVNT vertOut;
				vertOut.v = vertIn.v;
				vertOut.n = vertIn.n;
				vertOut.uv = vertIn.uv;
				fwrite(&vertOut, sizeof(VertexVNT), 1, fpOut);
			}
			break;
		case VNCT :
			{
				VertexVNCT vertOut;
				vertOut.v = vertIn.v;
				vertOut.n = vertIn.n;
				vertOut.c = vertIn.c;
				vertOut.uv = vertIn.uv;
				fwrite(&vertOut, sizeof(VertexVNCT), 1, fpOut);
			}
			break;
		}
	}

	fclose(fpIn);
	fclose(fpOut);
	return 1;
}
