#ifndef MESHREADER_H
#define MESHREADER_H

#ifndef DAVIS
#include "Point.h"
#else
#include <Geometry/Point.h>
#endif
#include <vector>
#include <stdio.h>
#include "ply.h"
#include "Vector3.h"
#include "rgb.h"
#include "Vertex.h"


/** this class is the interface for reading a mesh */
class AbstMeshReader
{
public:
	/// constructor
	AbstMeshReader() {}
	/// virtual destructor
	virtual ~AbstMeshReader() {}
	/// check if next item is a vertex
	virtual bool haveVertex() = 0;
	/// read the vertex
	virtual Vertex readVertex() = 0;
	

	/// check if next item is a face
	virtual bool haveFace() const = 0;
	/// read the face and append the vertex indices to vis
	virtual void readFace(STD vector<int>& vis) = 0;
	/// jump to the beginning of the file again
	virtual void restart() = 0;
	/// skip the vertices
	virtual void skipVertices();
};

/** implements a file based implementation */
class MeshReader : public AbstMeshReader
{
protected:
	/// store the filename
	string filename;
	/// store the open mode
	string mode;
	/// the input buffer
	FILE* fp;
	/// line buffer
	char buffer[4096];
public:
	/// constructor for given filename
	MeshReader(const string& filename, const string& mode = "rt");
	/// check if next item is a vertex
	bool haveVertex() { return false; }
	/// read the vertex
	Vertex readVertex() { 
		Vertex vert;
		vert.v.e[0] = vert.v.e[1] = vert.v.e[2] = 0;
		return vert; 
	}
	/// check if next item is a face
	bool haveFace() const { return false; }
	/// read the face and append the vertex indices to vis
	void readFace(STD vector<int>& vis) {}
	/// jump to the beginning of the file again
	void restart();
	/// open the file
	void open();
	/// return, whether we reached end of file
	bool eof() const;
	/// read the next line
	void getline();
};

/** reads from an obj file*/
class ObjReader : public MeshReader
{
protected:
	/// the index of the first vertex
	int minus;
	/// whether we have texture indices
	bool haveTextures;
	/// whether we have normal indices
	bool haveNormals;
	/// scan the next line
	void scanLine();
public:
	/// constructor for given filename
	ObjReader(const string& filename);
	/// check if next item is a vertex
	bool haveVertex();
	/// read the vertex
	Vertex readVertex();



	/// check if next item is a face
	bool haveFace() const;
	/// read the face and append the vertex indices to vis
	void readFace(STD vector<int>& vis);
	/// jump to the beginning of the file again
	void restart();
};


/** reads from a ply file*/
class PlyReader : public AbstMeshReader
{
protected:
	/// the ply file
	PlyFile *plyFile;
	/// store file name
	string fileName;
	/// type of element
	int elementType;
	/// the number of still to be read elements
	int nrElementsLeft;
	/// whether we are reading vertices
	bool haveVertices;
	/// whether we have vertex colors
	bool hasVertexColors;
	/// whether we have vertex normals
	bool hasVertexNormals;
	/// whether we have vertex textures
	bool hasVertexTextures;
	/// whether we have vertex material
	bool hasVertexMat;
	/// whether we have file index
	bool hasFileIdx;
	/// step after an element was read
	void stepElement();
	/// check the next element block
	void determineElementType();
	/// open the file
	void open();
	/// return, whether we reached end of file
	bool eof() const;
	/// close the ply file
	void close();
public:
	/// constructor for given filename
	PlyReader() : plyFile(0) {}
	/// constructor for given filename
	PlyReader(const string& name);
	/// initialize a reader constructed with the standard constructor
	void init(const string& name);
	/// destruct ply file
	~PlyReader();
	/// check if next item is a vertex
	bool haveVertex();
	/// read the vertex
	Vertex readVertex();
	Vertex readVertexWithColor(rgb &color);
	Vertex readVertexWithColorMatFile(rgb &color, int &mat, int &file);
	
	bool hasColor() const;
	bool hasVertNormal() const;
	bool hasVertTexture() const;
	bool hasVertMaterial() const;
	bool hasFileIndex() const;
	/// check if next item is a face
	bool haveFace() const;
	/// read the face and append the vertex indices to vis
	void readFace(STD vector<int>& vis);
	/// jump to the beginning of the file again
	void restart();
	/// skip the vertices
	void skipVertices();
	/// return the number of vertices
	int getNrVertices();
	/// return the number of faces
	int getNrFaces();
};

/** reads from a ply file*/
class PlySequenceReader : public AbstMeshReader
{
protected:
	/// base file name
	string baseFileName;
	/// for each file a ply files
	PlyReader* readers;
	/// store the range
	int from, to;
	/// store current reader
	int readerIndex, li, gi, ii;
	/// 
	struct IntPair
	{
		int li;
		int gi;
		bool operator < (const IntPair& ip) const { return li < ip.li; } 
	};
	/// between each two files the matches
	STD vector<IntPair>* indexMaps;
	/// between each two files the offset
	STD vector<int> offsets;
	/// return the match file name
	string getMatchName(int from) const { return baseFileName+"_"+toString(from)+"_"+toString(from+1)+".matches.txt"; }
	/// return a file name
	string getFileName(int index) const	{ return baseFileName+"_"+toString(index)+".ply.gz"; }
	/// initialize the read process
	void initRead();
	/// look for an index
	const IntPair& findIndex(int vi, int index) const;
	/// translate a reader index
	void translate(int& vi, int index) const;
public:
	/// constructor for given base file name and range of files
	PlySequenceReader(const string& baseName, int _from, int _to);
	/// destructor
	~PlySequenceReader();
	/// check if next item is a vertex
	bool haveVertex();
	/// read the vertex
	Vertex readVertex();
	/// check if next item is a face
	bool haveFace() const;
	/// read the face and append the vertex indices to vis
	void readFace(STD vector<int>& vis);
	/// jump to the beginning of the file again
	void restart();
};

#endif