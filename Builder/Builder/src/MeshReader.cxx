#include "Vector3.h"
#include "MeshReader.h"
#include "Files.h"
#include "Progression.h"
#include "ply.h"

#include <fstream>

//template class MeshReader<float>;
//template class ObjReader<float>;
//template class PlyReader<float>;
//template class PlySequenceReader<float>;

#include "common.h"

using namespace std;
	/// skip the vertices

void AbstMeshReader::skipVertices()
{
	while (haveVertex()) readVertex();
}

/// constructor for given filename

MeshReader::MeshReader(const string& file, const string& _mode) : filename(file), mode(_mode)
{
	open();
}


/// open the file

void MeshReader::open()
{
	if (*(filename.end()-1) == 'z')
		fp = fopenGzipped(filename.c_str(),mode.c_str());
	else
		fp = fopen(filename.c_str(),mode.c_str());
	IFWARN(fp == NULL, 0, "failed to open input stream") return;
}

/// jump to the beginning of the file again

void MeshReader::restart()
{
	fclose(fp);
	open();
}

/// return, whether we reached end of file

bool MeshReader::eof() const
{
	return feof(fp);
}

/// read the next line

void MeshReader::getline()
{
	fgets(&buffer[0], 4096, fp);
}

/// constructor for given filename

ObjReader::ObjReader(const string& filename) : MeshReader(filename)
{
	minus = 1;
	haveNormals = haveTextures = false;
	scanLine();
}

/// scan the next line

void ObjReader::scanLine()
{
	bool found = false;
	while (!found && !eof()) {
		getline();
		switch (buffer[0]) {
		case 'm' :
			sscanf(buffer, "minus %d", &minus);
			break;
		case 'v' :
			switch (buffer[1]) {
			case ' ' : 
				found = true;
				break;
			case 'n' :
				haveNormals = true;
				break;
			case 't' :
				haveTextures = true;
				break;
			}
			break;
		case 'f' :
			found = true;
			break;

		}
	}
}

/// check if next item is a vertex

bool ObjReader::haveVertex()
{
	return !eof() && (buffer[0] == 'v') && (buffer[1] == ' ');
}

/// read the vertex

Vertex ObjReader::readVertex()
{
	float x,y,z;
	sscanf(buffer+2, "%f %f %f", &x, &y, &z);
	scanLine();
	Vertex vert;
	vert.v.e[0] = x;
	vert.v.e[1] = y;
	vert.v.e[2] = z;
	return vert;
}


/// check if next item is a face

bool ObjReader::haveFace() const
{
	return !eof() && (buffer[0] == 'f');
}

bool scanNumber(const char* buffer, int& pos, const char* fmt, void* v)
{
	int tmp = pos;
	while (buffer[pos] &&  isspace(buffer[pos])) ++pos;
	while (buffer[pos] && !isspace(buffer[pos])) ++pos;
	return 1 == sscanf(&buffer[tmp], fmt, v);
}

/// read the face and append the vertex indices to vis

void ObjReader::readFace(STD vector<int>& vis)
{
	vis.clear();
	int n     = 0;
	int pos   = 2;
	int i     = 1;
	while (buffer[i] != 0) {
		if (buffer[i] == '/') buffer[i] = ' ';
		++i;
	}
	bool found = true;
	while (found) {
		int vi, ti, ni;
		found = scanNumber(buffer, pos, "%d", &vi); vi -= minus;
		found = found && ( !haveTextures || scanNumber(buffer, pos, "%d", &ti) );
		found = found && ( !haveNormals || scanNumber(buffer, pos, "%d", &ni) );
		if (found) vis.push_back(vi);
	}
	scanLine();
}

/// jump to the beginning of the file again

void ObjReader::restart()
{
	MeshReader::restart();
	scanLine();
}

typedef struct PlyVertex {
  float x,y,z;
  float nx, ny, nz;
  unsigned char r, g, b;
  float s, t, d;
  int mat, file;
} PlyVertex;

static PlyProperty vertex_props[] = { /* list of property information for a vertex */
  {"x", Float32, Float32, offsetof(PlyVertex,x), 0, 0, 0, 0},
  {"y", Float32, Float32, offsetof(PlyVertex,y), 0, 0, 0, 0},
  {"z", Float32, Float32, offsetof(PlyVertex,z), 0, 0, 0, 0},
  
  {"red", Uint8, Uint8, offsetof(PlyVertex,r), 0, 0, 0, 0},
  {"green", Uint8, Uint8, offsetof(PlyVertex,g), 0, 0, 0, 0},
  {"blue", Uint8, Uint8, offsetof(PlyVertex,b), 0, 0, 0, 0},

  {"nx", Float32, Float32, offsetof(PlyVertex,nx), 0, 0, 0, 0},
  {"ny", Float32, Float32, offsetof(PlyVertex,ny), 0, 0, 0, 0},
  {"nz", Float32, Float32, offsetof(PlyVertex,nz), 0, 0, 0, 0},

  {"s", Float32, Float32, offsetof(PlyVertex,s), 0, 0, 0, 0},
  {"t", Float32, Float32, offsetof(PlyVertex,t), 0, 0, 0, 0},
  {"d", Float32, Float32, offsetof(PlyVertex,d), 0, 0, 0, 0},

  {"mat", Uint32, Uint32, offsetof(PlyVertex,mat), 0, 0, 0, 0},
  {"file", Uint32, Uint32, offsetof(PlyVertex,file), 0, 0, 0, 0},
};

typedef struct PlyFace {
  unsigned char nverts;
  int *verts;
} PlyFace;


static PlyProperty face_props[] = { /* list of property information for a face */
  {"vertex_indices", Int32, Int32, offsetof(PlyFace,verts), 1, Uint8, Uint8, offsetof(PlyFace,nverts)},
};

/// constructor for given filename

PlyReader::PlyReader(const string& name)
{
	plyFile = 0;
	init(name);
}

/// constructor for given filename

void PlyReader::init(const string& name)
{
	fileName = name;
	open();
}

/// step after an element was read

void PlyReader::stepElement()
{
	if (--nrElementsLeft == 0) {
		++elementType;
		determineElementType();
	}
}

/// open

void PlyReader::open()
{
	elementType = 0;
	plyFile = 0;
	if (*(fileName.end()-1) == 'z') {
		FILE* fp = fopenGzipped(fileName.c_str(),"rb");
		IFWARN(fp == 0, 0, "cannot open file " << fileName << " for read.") return;
		plyFile = read_ply(fp);
		IFWARN(plyFile == 0, 0, "corrupt ply file for read.") return;
		if (plyFile->file_type == PLY_ASCII) {
			close();
			fp = fopenGzipped(fileName.c_str(),"rt");
			plyFile = read_ply(fp);
		}
	}
	else {
		char *tmpName = new char[fileName.length()+1];
		memcpy(tmpName, fileName.c_str(), fileName.length()+1);
		plyFile = open_ply_for_read(tmpName);
		delete [] tmpName;
	}
	determineElementType();
}

/// close the ply file

void PlyReader::close()
{
	close_ply (plyFile);
	free_ply (plyFile);
	plyFile = 0;
}

/// check the next element block

bool PlyReader::eof() const
{
	return elementType >= plyFile->num_elem_types;
}

/// step after an element was read

void PlyReader::skipVertices()
{
	Progression prog("skipping vertices", nrElementsLeft, 20);
	while (haveVertex()) {
		readVertex();
		prog.step();
	}
/*	if (haveVertex()) {
		++elementType;
		determineElementType();
	}*/
}


int PlyReader::getNrVertices()
{
	for (int et = 0; et < plyFile->num_elem_types; ++et) {
		int nr;
		char* elem_name = setup_element_read_ply (plyFile, et, &nr);
		if (equal_strings("vertex", elem_name)) return nr;
	}
	return -1;
}


int PlyReader::getNrFaces()
{
	for (int et = 0; et < plyFile->num_elem_types; ++et) {
		int nr;
		char* elem_name = setup_element_read_ply (plyFile, et, &nr);
		if (equal_strings("face", elem_name)) return nr;
	}
	return -1;
}

#include "App.h"

/// check the next element block

void PlyReader::determineElementType()
{
	int num_elems, nprops;
	PlyProperty **plist;	

	hasVertexColors = false;	// initialize
	hasVertexNormals = false;
	hasVertexTextures = false;
	hasVertexMat = false;
	hasFileIdx = false;

	if (eof()) return;
	do {
		char* elem_name = setup_element_read_ply (plyFile, elementType, &nrElementsLeft);
		plist = get_element_description_ply(plyFile, elem_name, &num_elems, &nprops);	

		if (equal_strings("vertex", elem_name)) {
			haveVertices = true;
			setup_property_ply(plyFile, &vertex_props[0]);
			setup_property_ply(plyFile, &vertex_props[1]);
			setup_property_ply(plyFile, &vertex_props[2]);

			for (int j=0; j<nprops; j++)
			{
				if (equal_strings("red", plist[j]->name))
				{
					setup_property_ply(plyFile, &vertex_props[3]);
					hasVertexColors = true;
				}
				else if (equal_strings("green", plist[j]->name))
				{
					setup_property_ply(plyFile, &vertex_props[4]);
					hasVertexColors = true;
				}
				else if (equal_strings("blue", plist[j]->name))
				{
					setup_property_ply(plyFile, &vertex_props[5]);
					hasVertexColors = true;
				}
				else if (equal_strings("nx", plist[j]->name))
				{
					setup_property_ply (plyFile, &vertex_props[6]);
					hasVertexNormals = true;
				}
				else if (equal_strings("ny", plist[j]->name))
				{
					setup_property_ply (plyFile, &vertex_props[7]);
					hasVertexNormals = true;
				}
				else if (equal_strings("nz", plist[j]->name))
				{
					setup_property_ply (plyFile, &vertex_props[8]);
					hasVertexNormals = true;
				}
				else if (equal_strings("s", plist[j]->name))
				{
					setup_property_ply (plyFile, &vertex_props[9]);
					hasVertexTextures = true;
				}
				else if (equal_strings("t", plist[j]->name))
				{
					setup_property_ply (plyFile, &vertex_props[10]);
					hasVertexTextures = true;
				}
				else if (equal_strings("d", plist[j]->name))
				{
					setup_property_ply (plyFile, &vertex_props[11]);
					hasVertexTextures = true;
				}
				else if (equal_strings("mat", plist[j]->name))
				{
					setup_property_ply (plyFile, &vertex_props[12]);
					hasVertexMat = true;
				}
				else if (equal_strings("file", plist[j]->name))
				{
					setup_property_ply (plyFile, &vertex_props[13]);
					hasFileIdx = true;
				}
			} 	


			/*
			if (hasVertexColors)
				cout << "PLY has vertex colors." << endl;
			if (hasVertexNormals)
				cout << "PLY has vertex normals." << endl;
			if (hasVertexTextures)
				cout << "PLY has vertex textures." << endl;
			*/
 
			break;
		}
		if (equal_strings ("face", elem_name)) {
			haveVertices = false;
			setup_property_ply (plyFile, &face_props[0]);
			break;
		}
		++elementType;
	} while (!eof());
}

/// destruct ply file

PlyReader::~PlyReader()
{
	close();
}

/// check if next item is a vertex

bool PlyReader::haveVertex()
{
	return !eof() && haveVertices;
}

/// check if model has vertex colors

bool PlyReader::hasColor() const
{
	return hasVertexColors;
}

bool PlyReader::hasVertNormal() const
{
	return hasVertexNormals;
}

bool PlyReader::hasVertTexture() const
{
	return hasVertexTextures;
}

bool PlyReader::hasVertMaterial() const
{
	return hasVertexMat;
}

bool PlyReader::hasFileIndex() const
{
	return hasFileIdx;
}
/// read the vertex

Vertex PlyReader::readVertex()
{
	PlyVertex vertex;
	get_element_ply(plyFile, (void *)&vertex);
	Vertex vert;
	vert.v.e[0] = vertex.x;
	vert.v.e[1] = vertex.y;
	vert.v.e[2] = vertex.z;
	#ifdef NORMAL_INCLUDED
	if(hasVertexNormals)
	{
		vert.n.e[0] = vertex.nx;
		vert.n.e[1] = vertex.ny;
		vert.n.e[2] = vertex.nz;
	}
	else
	{
		vert.n.e[0] = 0.0f;
		vert.n.e[1] = 0.0f;
		vert.n.e[2] = 0.0f;
	}
	#endif
	#ifdef COLOR_INCLUDED
	if(hasVertexColors)
	{
		vert.c.e[0] = vertex.r / 255.0f;
		vert.c.e[1] = vertex.g / 255.0f;
		vert.c.e[2] = vertex.b / 255.0f;
	}
	else
	{
		vert.c.e[0] = 1.0f;
		vert.c.e[1] = 1.0f;
		vert.c.e[2] = 1.0f;
	}
	#endif
	#ifdef TEXTURE_INCLUDED
	if(hasVertexTextures)
	{
		vert.uv.e[0] = vertex.s;
		vert.uv.e[1] = vertex.t;
	}
	#endif

	stepElement();

	return vert;
} 

/// read the vertex
Vertex PlyReader::readVertexWithColor(rgb &color)
{
	PlyVertex vertex;
	get_element_ply(plyFile, (void *)&vertex);

//	#ifdef DE_MODEL 
	// color changes
	/*
	75 14 188     100 90 130    // some purple thing
	xa0 x46 xbe    175 132 89  // deck floor
	38  203 30     80  170  90  // deck pipes
	x29 xdc xd1    80 90 150    // cyan crane
	*/
		
	if (vertex.r  == 175 &&
		vertex.g  == 14 &&
		vertex.b  == 188) {
		//printf ("hey\n");
		vertex.r  = 100;
		vertex.g  = 90;
		vertex.b  = 130;
	}
	if (vertex.r  == 160 &&
		vertex.g  == 70 &&
		vertex.b  == 190) {
		
		vertex.r  = 175;
		vertex.g  = 132;
		vertex.b  = 89;
	}
	if (vertex.r == 38 && 
		vertex.g  == 203 && 
		vertex.b  == 30) {
		//printf ("hey\n");
		vertex.r  = 80;
		vertex.g  = 170;
		vertex.b  = 90;
	}
	if (vertex.r  == 41 && 
		vertex.g  == 220 && 
		vertex.b  == 209) {
		//printf ("hey\n");
		vertex.r  = 80;
		vertex.g  = 90;
		vertex.b  = 150;
	}
//	#endif


	color[0] = vertex.r / 255.0f;
	color[1] = vertex.g / 255.0f;
	color[2] = vertex.b / 255.0f;

	Vertex vert;
	vert.v.e[0] = vertex.x;
	vert.v.e[1] = vertex.y;
	vert.v.e[2] = vertex.z;
	#ifdef NORMAL_INCLUDED
	if(hasVertexNormals)
	{
		vert.n.e[0] = vertex.nx;
		vert.n.e[1] = vertex.ny;
		vert.n.e[2] = vertex.nz;
	}
	else
	{
		vert.n.e[0] = 0.0f;
		vert.n.e[1] = 0.0f;
		vert.n.e[2] = 0.0f;
	}
	#endif
	#ifdef COLOR_INCLUDED
	if(hasVertexColors)
	{
		vert.c.e[0] = vertex.r / 255.0f;
		vert.c.e[1] = vertex.g / 255.0f;
		vert.c.e[2] = vertex.b / 255.0f;
	}
	else
	{
		vert.c.e[0] = 1.0f;
		vert.c.e[1] = 1.0f;
		vert.c.e[2] = 1.0f;
	}
	#endif
	#ifdef TEXTURE_INCLUDED
	if(hasVertexTextures)
	{
		vert.uv.e[0] = vertex.s;
		vert.uv.e[1] = vertex.t;
	}
	#endif
	stepElement();

	return vert;
}

/// read the vertex
Vertex PlyReader::readVertexWithColorMatFile(rgb &color, int &mat, int &file)
{
	PlyVertex vertex;
	get_element_ply(plyFile, (void *)&vertex);

//	#ifdef DE_MODEL 
	// color changes
	/*
	75 14 188     100 90 130    // some purple thing
	xa0 x46 xbe    175 132 89  // deck floor
	38  203 30     80  170  90  // deck pipes
	x29 xdc xd1    80 90 150    // cyan crane
	*/
		
	if (vertex.r  == 175 &&
		vertex.g  == 14 &&
		vertex.b  == 188) {
		//printf ("hey\n");
		vertex.r  = 100;
		vertex.g  = 90;
		vertex.b  = 130;
	}
	if (vertex.r  == 160 &&
		vertex.g  == 70 &&
		vertex.b  == 190) {
		
		vertex.r  = 175;
		vertex.g  = 132;
		vertex.b  = 89;
	}
	if (vertex.r == 38 && 
		vertex.g  == 203 && 
		vertex.b  == 30) {
		//printf ("hey\n");
		vertex.r  = 80;
		vertex.g  = 170;
		vertex.b  = 90;
	}
	if (vertex.r  == 41 && 
		vertex.g  == 220 && 
		vertex.b  == 209) {
		//printf ("hey\n");
		vertex.r  = 80;
		vertex.g  = 90;
		vertex.b  = 150;
	}
//	#endif


	color[0] = vertex.r / 255.0f;
	color[1] = vertex.g / 255.0f;
	color[2] = vertex.b / 255.0f;

	Vertex vert;
	vert.v.e[0] = vertex.x;
	vert.v.e[1] = vertex.y;
	vert.v.e[2] = vertex.z;
	#ifdef NORMAL_INCLUDED
	if(hasVertexNormals)
	{
		vert.n.e[0] = vertex.nx;
		vert.n.e[1] = vertex.ny;
		vert.n.e[2] = vertex.nz;
	}
	else
	{
		vert.n.e[0] = 0.0f;
		vert.n.e[1] = 0.0f;
		vert.n.e[2] = 0.0f;
	}
	#endif
	#ifdef COLOR_INCLUDED
	if(hasVertexColors)
	{
		vert.c.e[0] = vertex.r / 255.0f;
		vert.c.e[1] = vertex.g / 255.0f;
		vert.c.e[2] = vertex.b / 255.0f;
	}
	else
	{
		vert.c.e[0] = 1.0f;
		vert.c.e[1] = 1.0f;
		vert.c.e[2] = 1.0f;
	}
	#endif
	#ifdef TEXTURE_INCLUDED
	if(hasVertexTextures)
	{
		vert.uv.e[0] = vertex.s;
		vert.uv.e[1] = vertex.t;
	}
	#endif
	if(hasVertexMat)
	{
		mat = vertex.mat;
	}
	if(hasFileIdx)
	{
		file = vertex.file;
	}
	stepElement();

	return vert;
}
/// check if next item is a face

bool PlyReader::haveFace() const
{
	return !eof() && !haveVertices;
}

/// read the face and append the vertex indices to vis

void PlyReader::readFace(STD vector<int>& vis)
{
	PlyFace face;
	get_element_ply(plyFile, (void *) &face);
	stepElement();
	vis.resize(face.nverts);
	for (int i=0; i<face.nverts; ++i) vis[i] = face.verts[i];
	free(face.verts);
}

/// jump to the beginning of the file again

void PlyReader::restart()
{
	close();
	open();
}


/// initialize the read process

void PlySequenceReader::initRead()
{
	gi = 0;
	li = 0;
	ii = 0;
	readerIndex = from;
}
/// look for an index

const PlySequenceReader::IntPair& PlySequenceReader::findIndex(int vi, int index) const
{
	const STD vector<IntPair>& indexMap = indexMaps[index-from-1];
	STD vector<IntPair>::const_iterator left  = indexMap.begin();
	STD vector<IntPair>::const_iterator right = indexMap.end()-1;
	// if vertex is in first or last segment, return int pair
	if ( (*right).li <= vi ) return *right;
	if ( (*left).li >= vi ) return *left;
	// otherwise search recursively
	while (right-left > 1) {
		STD vector<IntPair>::const_iterator middle = left + (right-left)/2;
		int li = (*middle).li;
		if (li == vi) return *middle;
		if (li < vi) left = middle;
		else right = middle;
	}
	// finally return the int pair
	return *left;
}
/// translate a reader index

void PlySequenceReader::translate(int& vi, int index) const
{
	// in first file nothing is changed
	if (index == from) return;
	// search the index in the corresponding map
	const IntPair& ip = findIndex(vi, index);
	// in case of an internal index
	if (ip.gi < 0) 
		// update index
		vi = vi + offsets[index-from-1] + ip.gi + 1;
	else {
		// otherwise translate index of previous file
		vi = ip.gi;
		translate(vi, index - 1);
	}
}

/// constructor for given base file name and range of files

PlySequenceReader::PlySequenceReader(const string& baseName, int _from, int _to)
{
	// set parameters
	baseFileName = baseName;
	from = _from;
	to = _to;
	// compute index maps
	int i;
	char buffer[257];
	// between each two files allocate a mapping
	indexMaps = new STD vector<IntPair>[to-from];
	offsets.resize(to-from);
	// for each mapping
	for (i=from; i<to; ++i) {
		// uninitialize offset
		offsets[i-from] = -1;
		// initialize interval border to relative offset minus one
		IntPair ipl;
		ipl.li = 0;
		ipl.gi = -1;
		// open match file
		STD ifstream is(getMatchName(i).c_str());
		IFWARN(is.fail(), 0, "could not open match file " << getMatchName(i)) continue;
		// skip header
		for (int j=0; j<3; ++j) {
			is.getline(&buffer[0],256);
			PRINTLN(buffer);
		}
		// read pairs of indices till end of file
		while (!is.eof()) {
			IntPair ip;
			is >> ip.gi >> ip.li;
			if (is.fail()) break;
			IFWARN(ip.li < ipl.li, 0, "non increasing sequence found: " << ipl.li << "->" << ip.li) {}
			// for non successive indices insert an interval marker at the beginning of the interval
			if ( ip.li != ipl.li) indexMaps[i-from].push_back(ipl);
			// insert the mapping
			indexMaps[i-from].push_back(ip);
			// set the next interval marker
			ipl.li  = ip.li+1;
			ipl.gi -= 1;
		}
		// at the end add an intervall marker
		indexMaps[i-from].push_back(ipl);
		PRINTLN("map file " << i << " contains " << indexMaps[i-from].size() << " pairs");
	}
	// open all ply files at the same time
	readers = new PlyReader[to-from+1];
	for (i=from; i<=to; ++i) {
		readers[i-from].init(getFileName(i));
	}
	// set the first reader
	initRead();
}
/// destructor

PlySequenceReader::~PlySequenceReader()
{
	delete [] readers;
	delete [] indexMaps;
}
/// check if next item is a vertex

bool PlySequenceReader::haveVertex()
{
	if (!readers[readerIndex-from].haveVertex()) return false;
	bool skipVertex;
	bool have;
	do {
		// check if vertex is a new one
		skipVertex = false;
		if ( (readerIndex > from) && (ii != -1) ) {
			const IntPair ip = indexMaps[readerIndex-from-1][ii];
			if (ip.li == li) {
				if (ip.gi >= 0) {
					skipVertex = true;
					--gi;
					readVertex();
					//PRINTLN("skip vertex " << li << " = " << gi);
				}
				if (++ii == indexMaps[readerIndex-from-1].size()) ii = -1;
			}
		}
		have = readers[readerIndex-from].haveVertex();
	} while (skipVertex && have);
	return have;
}
/// read the vertex

//Vector3 PlySequenceReader::readVertex()
Vertex PlySequenceReader::readVertex()
{
	Vertex p = readers[readerIndex-from].readVertex();
	++li;
	++gi;
	if (!readers[readerIndex-from].haveVertex()) {
		offsets[readerIndex-from-1] = gi;
		PRINTLN("setting offset " << readerIndex-1 << " to " << gi);
		li = 0;
		ii = 0;
		if (++readerIndex > to) readerIndex = from;
	}
	return p;
}
/// check if next item is a face

bool PlySequenceReader::haveFace() const
{
	return readers[readerIndex-from].haveFace();
}
/// read the face and append the vertex indices to vis

void PlySequenceReader::readFace(STD vector<int>& vis)
{
	// read face with old indices
	readers[readerIndex-from].readFace(vis);
	// translate indices
	for (STD vector<int>::iterator iter = vis.begin(); iter != vis.end(); ++iter) {
		translate(*iter, readerIndex);
	}
	// check if reader index has to incremented
	if (!haveFace()) {
		if (++readerIndex > to) readerIndex = from;
	}
}
/// jump to the beginning of the file again

void PlySequenceReader::restart()
{
	for (int i=from; i<=to; ++i) readers[i-from].restart();
	initRead();
}
