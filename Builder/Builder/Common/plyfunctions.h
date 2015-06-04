#ifndef COMMON_PLYFUNCTIONS_H
#define COMMON_PLYFUNCTIONS_H

#include "Vector3.h"
#include "rgb.h"

enum PlyState {
	PLY_NOTLOADED = 0x01,
	PLY_OPENED = 0x02,
	PLY_HEADERPARSED = 0x03,
	PLY_READVERTICES = 0x04,
	PLY_READFACES = 0x05,
	PLY_FINISHED = 0x06,
	PLY_CLOSED = 0x0A,
	PLY_FOPENERROR = 0x07,
	PLY_HEADERERROR = 0x08,
	PLY_PARSEERROR = 0x09,
};

enum PlyFeature {
	PLY_COLOR = 0x01,
	PLY_NORMAL = 0x02
};

class PLYLoader
{
public:
	PLYLoader() {
		currentState = PLY_NOTLOADED;
		fp = NULL;
		vertexScanStr = NULL;
		faceScanStr = NULL;

		b_hasColor = false;
		b_hasNormal = false;
	}

	~PLYLoader() {
		if (currentState != PLY_CLOSED && 
			currentState != PLY_FOPENERROR &&
			currentState != PLY_NOTLOADED &&
			fp != NULL)
			fclose(fp);
	}

	bool plyOpen(const char *filename);

	bool plyParseHeader();
	void plyClose();

	bool plyHasFeature(PlyFeature query);
	unsigned int plyGetNumVertices();
	unsigned int plyGetNumFaces();

	bool plyNextVertex();
	void plyGetVertex(Vector3 &dest);
	void plyGetVertexColor(rgb &dest);
	void plyGetVertexNormal(Vector3 &dest);

	bool plyNextFace();
	void plyGetFaceIndices(unsigned int *dest);

protected:

	PlyState currentState; 
	FILE *fp;

	char *vertexScanStr;
	char *faceScanStr;

	// feature flags:
	bool b_hasColor;
	bool b_hasNormal;
	
private:
};


#endif