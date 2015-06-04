#ifndef OCTREE_H
#define OCTREE_H

#include "Voxel.h"
#include "BV.h"
#include <hash_map>

#define MAX_DIM (__int64)4096

class Octree
{
public:
	typedef struct Position_t
	{
		int x, y, z;

		Position_t(int x, int y, int z) : x(x), y(y), z(z) {}

		operator size_t() const {return x + y * MAX_DIM + z * MAX_DIM * MAX_DIM;}

		bool operator == (const Position_t & p)
		{
			if(x != p.x) return false;
			if(y != p.y) return false;
			return z == p.z;
		}

		bool operator != (const Position_t & p)
		{
			return !(*this == p);
		}
	} Position;

	struct PositionLess
	{
		bool operator() (const Position& a, const Position& b) const
		{
			return a.x + a.y * MAX_DIM + a.z * MAX_DIM * MAX_DIM > b.x + b.y * MAX_DIM + b.z * MAX_DIM * MAX_DIM;
		}
	};

public:
	OctreeHeader m_header;
	Voxel *m_octree;
	int m_numVoxels;
	int m_voxelSize;
	Vector3 m_voxelDelta;

	typedef stdext::hash_map<Position, int, std::hash_compare<Position, PositionLess> > Hash;
	Hash m_hash;

	AABB computeSubBox(int x, int y, int z, const AABB &box);
	AABB computeBox(const Position &pos);

	void buildHash(const AABB &bb, int index, int depth);
	void buildHash(int startDepth = 1);
public:
	Octree(void);
	~Octree(void);

	Octree(const OctreeHeader& header, Voxel *octree, int numVoxels);
	Octree(const char* fileName);

	void load(const OctreeHeader& header, Voxel *octree, int numVoxels);
	void load(const char* fileName, int startDepth = 1);
	void save(const char* fileName);

	const OctreeHeader &getHeader() {return m_header;}
	int getNumVoxels() {return m_numVoxels;}
	Voxel *getOctreePtr() {return m_octree;}
	Voxel& operator [] (int i) {return m_octree[i];}

	Position getPosition(const Vector3& pos);
	Position getPosition(const AABB &bb);
};

#endif // OCTREE_H