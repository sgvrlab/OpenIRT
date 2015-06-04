#include "stdafx.h"
#include "common.h"

using namespace std;

TextureManager* TextureManager::m_Singleton = 0;

TextureManager::TextureManager() {
	lastError[0] = 0;
}

TextureManager::~TextureManager() {
	clear();
	if (m_Singleton != 0)
		delete m_Singleton;
}

BitmapTexture* TextureManager::loadTexture(const char *name, const char *basePath) {
	BitmapTexture *texPtr;
	if (name == NULL)
		return NULL;

	// look whether this texture is already loaded:
	//texPtr = m_Textures[name];
	TextureList::const_iterator it = m_Textures.find(name);
	if (it != m_Textures.end()) { // if yes, increase its reference counter and return it
		texPtr = it->second;
		texPtr->refCount++;
		return texPtr;
	}

	char *name2 = new char[strlen(name)+1];
	strcpy(name2, name);

	// else: create a new one
	texPtr = new BitmapTexture();

	if (texPtr == 0) {
		sprintf(lastError, "TextureManager: error creating new texture instance. Out of memory ?");
		return NULL;
	}

	if (texPtr->loadFromFile(name2, basePath)) {
		// enter into texture list
		texPtr->refCount++;		
		
		m_Textures[name2] = texPtr;
		return texPtr;
	}
	else { // error loading bitmap:
		sprintf(lastError, "TextureManager: error loading file '%s': %s", name, texPtr->getLastErrorString());
		return NULL;
	}
}

void TextureManager::unloadTexture(const char *name){
	BitmapTexture *texPtr;

	// look whether this texture is loaded:
	texPtr = m_Textures[name];
	if (texPtr) { // if yes, increase its reference counter and return it
		texPtr->refCount--;

		// if this is the last used reference, also delete the texture
		if (texPtr->refCount == 0) {
			delete texPtr;
			m_Textures.erase(name);
		}
	}
}

void TextureManager::unloadTexture(BitmapTexture *texPtr) {

	// look whether this texture is loaded:
	TextureList::iterator it = m_Textures.begin();
	TextureList::iterator itEnd = m_Textures.end();
	for (; it != itEnd; ++it)
	{
		// was found in hash_map
		if (it->second == texPtr) {
			texPtr->refCount--;

			// if this is the last used reference, also delete the texture
			if (texPtr->refCount == 0) {
				delete texPtr;
				m_Textures.erase(it);
			}

			return;
		}
	}
}

void TextureManager::clear() {

	// go through texture list
	for (TextureList::iterator it = m_Textures.begin(); it != m_Textures.end(); ++it)
	{
		// delete texture
		delete it->second;
	}

	// delete list
	m_Textures.clear();
}


TextureManager* TextureManager::getSingletonPtr()
{
	if (m_Singleton == 0)
		m_Singleton = new TextureManager();
	return m_Singleton;
}

TextureManager& TextureManager::getSingleton()
{  
	if (m_Singleton == 0)
		m_Singleton = new TextureManager();
	return *m_Singleton;
}