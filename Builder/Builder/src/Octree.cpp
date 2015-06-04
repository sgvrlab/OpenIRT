#include "Octree.h"
#include <io.h>

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

void Octree::load(const char* fileName, int startDepth)
{
	FILE *fp;
	fopen_s(&fp, fileName, "rb");
	int fileSize = _filelength(_fileno(fp)) - sizeof(OctreeHeader);
	m_numVoxels = fileSize / sizeof(Voxel);

	if(m_octree) delete[] m_octree;
	m_octree = new Voxel[m_numVoxels];
	fread(&m_header, sizeof(OctreeHeader), 1, fp);
	fread(m_octree, fileSize, 1, fp);
	fclose(fp);

	buildHash(startDepth);
}

void Octree::save(const char* fileName)
{
	FILE *fp;
	fopen_s(&fp, fileName, "wb");
	fwrite(&m_header, sizeof(OctreeHeader), 1, fp);
	fwrite(m_octree, m_numVoxels*sizeof(Voxel), 1, fp);
	fclose(fp);
}

AABB Octree::computeSubBox(int x, int y, int z, const AABB &box)
{
	AABB subBox;
	float boxDelta = (box.max.x() - box.min.x()) / m_header.dim;
	subBox.min = box.min + (Vector3((float)x, (float)y, (float)z)*boxDelta);
	subBox.max = subBox.min + Vector3(boxDelta);
	return subBox;
}

AABB Octree::computeBox(const Position &pos)
{
	AABB bb;
	bb.min.e[0] = m_header.min.e[0] + m_voxelDelta.e[0] * pos.x;
	bb.min.e[1] = m_header.min.e[1] + m_voxelDelta.e[1] * pos.y;
	bb.min.e[2] = m_header.min.e[2] + m_voxelDelta.e[2] * pos.z;
	bb.max = bb.min + m_voxelDelta;
	return bb;
}

Octree::Position Octree::getPosition(const Vector3& pos)
{
	return Position((int)((pos[0] - m_header.min[0]) / m_voxelDelta[0]), (int)((pos[1] - m_header.min[1]) / m_voxelDelta[1]), (int)((pos[2] - m_header.min[2]) / m_voxelDelta[2]));
}
Octree::Position Octree::getPosition(const AABB &bb)
{
	return getPosition(bb.min + (0.5f*m_voxelDelta));
}

void Octree::buildHash(const AABB &bb, int index, int depth)
{
	int N = m_header.dim;
	int childIndex = index * N * N * N;

	for(int x=0;x<N;x++)
		for(int y=0;y<N;y++)
			for(int z=0;z<N;z++)
			{
				AABB subBox = computeSubBox(x, y, z, bb);
				//if(!m_octree[childIndex].hasChild())
				//{
				//	if(m_octree[childIndex].isLeaf())
				//	{
				//		m_hash[getPosition(subBox)]= childIndex;
				//	}
				//}
				if(depth == m_header.maxDepth && (m_octree[childIndex].isLeaf() || m_octree[childIndex].hasChild()))
				{
					m_hash[getPosition(subBox)] = childIndex;
				}
				else
				{
					if(m_octree[childIndex].hasChild())
						buildHash(subBox, m_octree[childIndex].getChildIndex(), depth+1);
				}
				childIndex++;
			}
}

void Octree::buildHash(int startDepth)
{
	m_voxelSize = m_header.dim << (m_header.maxDepth-startDepth);
	m_voxelDelta = (m_header.max - m_header.min) / (float)m_voxelSize;

	m_hash.clear();
	buildHash(AABB(m_header.min, m_header.max), 0, startDepth);
}