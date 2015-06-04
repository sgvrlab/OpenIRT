#include "defines.h"
#include "BitmapTexture.h"
#include "TextureManager.h"

using namespace irt;

TextureManager::TextureManager() {
	lastError[0] = 0;
}

TextureManager::~TextureManager() {
	clear();
}

BitmapTexture* TextureManager::loadTexture(const char *name) {
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

	std::string name2(name);

	// else: create a new one
	texPtr = new BitmapTexture();

	if (texPtr == 0) {
		sprintf_s(lastError, 500, "TextureManager: error creating new texture instance. Out of memory ?");
		return NULL;
	}

	if (texPtr->loadFromFile(name)) {
		// enter into texture list
		texPtr->refCount++;		
		
		m_Textures[name2] = texPtr;

		return texPtr;
	}
	else { // error loading bitmap:
		sprintf_s(lastError, 500, "TextureManager: error loading file '%s': %s", name, texPtr->getLastErrorString());

		delete texPtr;
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

void TextureManager::unloadTexture(BitmapTexture *texPtr) 
{
	if(!m_Textures.size()) return;

	// look whether this texture is loaded:
	TextureList::iterator it = m_Textures.begin();
	TextureList::iterator itEnd = m_Textures.end();
	for (; it != itEnd; ++it)
	{
		// was found in list
		if (it->second == texPtr) {
		//if (*it._Ptr == texPtr) {
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
		//delete *it._Ptr;
	}

	// delete list
	m_Textures.clear();
}


TextureManager* TextureManager::getSingletonPtr()
{
	static TextureManager manager;
	return &manager;
}