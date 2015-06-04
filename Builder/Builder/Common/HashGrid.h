#ifndef COMMON_HASHGRID_H
#define COMMON_HASHGRID_H

#include <hash_map>

/********************************************************************
	created:	2005/02/18
	created:	18:2:2005   16:48
	filename: 	c:\MSDev\MyProjects\Renderer\Common\HashGrid.h
	file path:	c:\MSDev\MyProjects\Renderer\Common
	file base:	HashGrid
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Grid structure using a hash table internally
*********************************************************************/

template <class T> class HashGrid
{
public:
	/**
	 * Normal constructor, creates empty grid with values of
	 * defaultValue for all grid points.
	 */
	HashGrid(unsigned int nx, unsigned int ny, unsigned int nz, T &defaultValue) {
		this->nx = nx;
		this->ny = ny;
		this->nz = nz;
		nxtimesny = nx*ny; // precalc offset

		defaultElement = defaultValue;
	}

	HashGrid() {
		nx = 1;
		ny = 1;
		nz = 1;
		nxtimesny = nx*ny; // precalc offset
	}

	/**
	 * Copy constructor
	 */
	HashGrid(const HashGrid &orig) {
		nx = orig.nx;
		ny = orig.ny;
		nz = orig.nz;
		nxtimesny = nx*ny; // precalc offset

		defaultElement = orig.defaultElement;

		m_Table = orig.m_Table;
	}

	void operator=(const HashGrid &right) {
		nx = right.nx;
		ny = right.ny;
		nz = right.nz;
		nxtimesny = nx*ny; // precalc offset

		defaultElement = right.defaultElement;

		m_Table = right.m_Table;
	}

	/**
	 * Dtor
	 */
	~HashGrid() {	
		clear();
	};

	/**
	 * Change dimensions
	 */
	void setDimensions(unsigned int nx, unsigned int ny, unsigned int nz) {
		clear();
		this->nx = nx;
		this->ny = ny;
		this->nz = nz;
		nxtimesny = nx*ny; // precalc offset
	}

	/**
	 * Set default value if grid point not set yet
	 */
	void setDefaultValue(T &defaultValue) {
		defaultElement = defaultValue;
	}

	/**
	 * Empties the hash table
	 */
	void clear() {		
		m_Table.clear();
	}

	/**
	 * Access operators:
	 */

	/**
	 * Write access via bracket operator
	 */
	T &operator[](unsigned int address) { 
		HashTable::iterator it = m_Table.find(address);
		if (it != m_Table.end())
			return (*it).second;
		
		m_Table[address] = defaultElement;
		return m_Table.find(address)->second;
	}

	/**
	 * Read access via bracket operator
	 */
	T &operator[](unsigned int address) const { 
		HashTable::iterator it = m_Table.find(address);
		if (it == m_Table->end())
			return defaultElement;
		else
			return (*it).second;
	}

	/**
	 * Read access via indicies
	 */
	T &at(unsigned int i, unsigned int j, unsigned int k) {
		HashTable::iterator it = m_Table.find(k*nxtimesny + j*nx + i);
		if (it == m_Table.end())
			return defaultElement;
		else
			return (*it).second;
	}

	/**
	* Read access via offset
	*/
	T &at(unsigned int address) {
		HashTable::iterator it = m_Table.find(address);
		if (it == m_Table.end())
			return defaultElement;
		else
			return (*it).second;
	}

	/**
	 * Write access via indices
	 */
	void set(unsigned int i, unsigned int j, unsigned int k, T &value) {
		m_Table[k*nxtimesny + j*nx + i] =  value;
	}

	/**
	 * Write access via offset
	*/
	void set(unsigned int address, T &value) {
		m_Table[address] =  value;
	}

	/**
	 * Addition via indices
	 */
	void add(unsigned int i, unsigned int j, unsigned int k, T &value) {		
		m_Table[k*nxtimesny + j*nx + i] += value;
	}

	/**
	 * Addition via offset
	 */
	void add(unsigned int address, T &value) {		
		m_Table[address] += value;
	}

	/**
	 * Deletion via indices
	 */
	void erase(unsigned int i, unsigned int j, unsigned int k) {
		m_Table.erase(k*nxtimesny + j*nx + i);
	}

	/**
	 * Deletion via offset
	 */
	void erase(unsigned int address) {
		m_Table.erase(address);
	}

	/**
	 * Output statistics on the hash
	 */
	void printStats(const char *LoggerName = NULL) {
		LogManager *log = LogManager::getSingletonPtr();
		char outputBuffer[2000];		
		sprintf(outputBuffer, "Used :\t%d of %d cells (%.1f%%)", m_Table.size(), nx*ny*nz, 
			   (float)(100.0f* m_Table.size() / (float)(nx*ny*nz)));
		log->logMessage(outputBuffer, LoggerName);		
		sprintf(outputBuffer, "Approx. mem usage :\t%d KB", (int)(m_Table.size()*sizeof(T)/1024.0f));//, 
															//(int)(100.0f* m_Table.size() / (nx*ny*nz)));
		log->logMessage(outputBuffer, LoggerName);		
	}

protected:

	// underlying hash table class (STLport hash table)
	typedef stdext::hash_map<unsigned int, T> HashTable;
	typedef std::pair<unsigned int, T> HashTableElement;

	// Hash table instance
	HashTable m_Table;

	// dimensions:
	unsigned int nx, ny, nz;
	unsigned int nxtimesny;

	// default value for grid points that have no value yet
	T defaultElement;
	
private:
};

#endif