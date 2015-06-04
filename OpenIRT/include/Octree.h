#pragma once

#include "Voxel.h"
#include "BV.h"
#include <hash_map>

#define MAX_DIM 1024

namespace irt
{

typedef struct OctreeHeader_t
{
	int dim;
	int maxDepth;
	Vector3 min;
	Vector3 max;
} OctreeHeader;

class Octree
{
protected:
	typedef struct Position_t
	{
		int x, y, z;

		Position_t(int x, int y, int z) : x(x), y(y), z(z) {}

		operator size_t() const {return x + y * MAX_DIM + z * MAX_DIM * MAX_DIM;}
	} Position;

	struct PositionLess
	{
		bool operator() (const Position& a, const Position& b) const
		{
			return a.x + a.y * MAX_DIM + a.z * MAX_DIM * MAX_DIM > b.x + b.y * MAX_DIM + b.z * MAX_DIM * MAX_DIM;
		}
	};

protected:
	OctreeHeader m_header;
	Voxel *m_octree;
	int m_numVoxels;
	int m_voxelSize;
	Vector3 m_voxelDelta;

	typedef stdext::hash_map<Position, int, std::hash_compare<Position, PositionLess> > Hash;
	Hash m_hash;

	AABB computeSubBox(int x, int y, int z, const AABB &box);

	void buildHash(const AABB &bb, int index);
	void buildHash();

public:
	Octree(void);
	~Octree(void);

	Octree(const OctreeHeader& header, Voxel *octree, int numVoxels);
	Octree(const char* fileName);

	void load(const OctreeHeader& header, Voxel *octree, int numVoxels);
	void load(const char* fileName, bool _buildHash = true);
	void load(FILE *fp, int numVoxels, const AABB &bb, bool _buildHash = true);

	OctreeHeader &getHeader() {return m_header;}
	int getNumVoxels() {return m_numVoxels;}
	float getLeafVoxelSize() {return m_voxelDelta.e[0];}
	Voxel *getOctreePtr() {return m_octree;}
	Voxel& operator [] (int i) {return m_octree[i];}

	Position getPosition(const Vector3& pos);
	Position getPosition(const AABB &bb);

	void setOctreePtr(Voxel *octree) {m_octree = octree;}
};

};