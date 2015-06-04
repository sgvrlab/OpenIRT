#include "Octree.h"
#include <io.h>

using namespace irt;

Octree::Octree(void)
	: m_octree(0)
{
}

Octree::~Octree(void)
{
	if(m_octree) delete[] m_octree;
}

Octree::Octree(const OctreeHeader& header, Voxel *octree, int numVoxels)
{
	load(header, octree, numVoxels);
}

Octree::Octree(const char* fileName)
{
	load(fileName);
}

void Octree::load(const OctreeHeader& header, Voxel *octree, int numVoxels)
{
	m_header = header;
	m_numVoxels = numVoxels;

	if(m_octree) delete[] m_octree;
	m_octree = new Voxel[numVoxels];
	memcpy(m_octree, octree, numVoxels*sizeof(Voxel));
}

void Octree::load(const char* fileName, bool _buildHash)
{
	FILE *fp;
	fopen_s(&fp, fileName, "rb");
	if(fp == NULL)
	{
		printf("Cannot open %s!\n", fileName);
	}
	int fileSize = _filelength(_fileno(fp)) - sizeof(OctreeHeader);
	m_numVoxels = fileSize / sizeof(Voxel);

	if(m_octree) delete[] m_octree;
	m_octree = new Voxel[m_numVoxels];
	fread(&m_header, sizeof(OctreeHeader), 1, fp);
	fread(m_octree, fileSize, 1, fp);
	fclose(fp);

	if(_buildHash) buildHash();

	m_voxelSize = m_header.dim << (m_header.maxDepth-1);
	m_voxelDelta = (m_header.max - m_header.min) / (float)m_voxelSize;
}

void Octree::load(FILE *fp, int numVoxels, const AABB &bb, bool _buildHash)
{
	m_numVoxels = numVoxels;

	if(m_octree) delete[] m_octree;
	m_octree = new Voxel[m_numVoxels];
	fread(m_octree, sizeof(Voxel), numVoxels, fp);
	m_header.min = bb.min;
	m_header.max = bb.max;

	if(_buildHash) buildHash();

	m_voxelSize = m_header.dim << (m_header.maxDepth-1);
	m_voxelDelta = (m_header.max - m_header.min) / (float)m_voxelSize;
}

AABB Octree::computeSubBox(int x, int y, int z, const AABB &box)
{
	AABB subBox;
	float boxDelta = (box.max.x() - box.min.x()) / m_header.dim;
	subBox.min = box.min + (Vector3((float)x, (float)y, (float)z)*boxDelta);
	subBox.max = subBox.min + Vector3(boxDelta);
	return subBox;
}

Octree::Position Octree::getPosition(const Vector3& pos)
{
	return Position((int)((pos[0] - m_header.min[0]) / m_voxelDelta[0]), (int)((pos[1] - m_header.min[1]) / m_voxelDelta[1]), (int)((pos[2] - m_header.min[2]) / m_voxelDelta[2]));
}
Octree::Position Octree::getPosition(const AABB &bb)
{
	return getPosition(bb.min + (0.5f*m_voxelDelta));
}

void Octree::buildHash(const AABB &bb, int index)
{
	int N = m_header.dim;
	int childIndex = index * N * N * N;

	for(int x=0;x<N;x++)
		for(int y=0;y<N;y++)
			for(int z=0;z<N;z++)
			{
				AABB subBox = computeSubBox(x, y, z, bb);
				if(!m_octree[childIndex].hasChild())
				{
					if(m_octree[childIndex].isLeaf())
					{
						m_hash[getPosition(subBox)] = childIndex;
					}
				}
				else
				{
					buildHash(subBox, m_octree[childIndex].getChildIndex());
				}
				childIndex++;
			}
}

void Octree::buildHash()
{
	m_hash.clear();
	buildHash(AABB(m_header.min, m_header.max), 0);
}