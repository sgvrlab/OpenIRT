#include "stdafx.h"
#include "plyfunctions.h"

bool PLYLoader::plyOpen(const char *filename) {
	return false;
}

bool PLYLoader::plyParseHeader() {
	return false;
}

void PLYLoader::plyClose() {
	if (fp)
		fclose(fp);
	currentState = PLY_CLOSED;
}

bool PLYLoader::plyHasFeature(PlyFeature query) {
	switch (query) {
		case PLY_COLOR:
			return b_hasColor;
		case PLY_NORMAL:
			return b_hasNormal;
		default:
			return false;
	}
}

unsigned int PLYLoader::plyGetNumVertices() {
	return 0;
}

unsigned int PLYLoader::plyGetNumFaces() {
	return 0;
}

bool PLYLoader::plyNextVertex() {
	return false;
}

void PLYLoader::plyGetVertex(Vector3 &dest) {
}

void PLYLoader::plyGetVertexColor(rgb &dest) {
}

void PLYLoader::plyGetVertexNormal(Vector3 &dest) {
}

bool PLYLoader::plyNextFace() {
	return false;
}

void PLYLoader::plyGetFaceIndices(unsigned int *dest) {

}