#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "OptionManager.h"
#include "BVHNodeDefine.h"
#include "Triangle.h"
#include <io.h>

#include "MTLGenerator.h"

#include "Progression.h"
#include "helpers.h"

int MTLGenerator::Do(const char* filepath)
{
	TimerValue start, end;
	start.set();

	char oocFileName[MAX_PATH];
	char mtlFileName[MAX_PATH];
	sprintf(oocFileName, "%s/materials.ooc", filepath);
	sprintf(mtlFileName, "%s/material.mtl", filepath);

	typedef struct OOCMaterial_t
	{
		char dummy[52];
		Vector3 color;
	} OOCMaterial;

	FILE *fpSrc, *fpMTL;
	if(fopen_s(&fpSrc, oocFileName, "rb") != 0) return false;

	int fileSize = (int)_filelength(_fileno(fpSrc));

	Progression prog("Generate MTL", fileSize/sizeof(OOCMaterial), 100);

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
		prog.step();
	}

	fclose(fpSrc);
	fclose(fpMTL);

	end.set();
	
	float elapsedHours;
	int elapsedMinOfHour;
	double elapsedMinOfHourFrac = modf((end - start)/(float)(60*60), &elapsedHours);
	elapsedMinOfHour = elapsedMinOfHourFrac * 60.0;
	
	cout << "Generating MTL file ended, time = " << (end - start) << "s (" << (int)elapsedHours << " h, " << elapsedMinOfHour << " min)" << endl;

	return 1;
}