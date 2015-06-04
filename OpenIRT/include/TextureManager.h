#pragma once

#include <map>
#include <string>
#include <hash_map>

struct eqstrTextureManager
{
	bool operator()(const char* s1, const char* s2) const
	{
		return strcmp(s1, s2) == 0;
	}
};

struct greater_strTextureManager {
   bool operator()(const char* x, const char* y) const {
      if ( strcmp(x, y) < 0)
         return true;

      return false;
   }
};

namespace irt
{

class TextureManager
{
public:
	

	/**
	 * Create a new Texture from an image file.
	 *
	 * Returns NULL on error, use getLastError() to get the
	 * error message. If basePath is set, then it will be used as the
	 * path for searching the texture instead of the system default (which
	 * is set in options.xml)
	 */
	BitmapTexture* loadTexture(const char *name);

	void unloadTexture(const char *name);
	void unloadTexture(BitmapTexture *texPtr);
	// TODO: load texture from memory

	/**
	 * Number of textures currently loaded.
	 */
	unsigned int getNumTextures() {
		return (unsigned int)m_Textures.size();
	}
	
	/**
	 * Deletes all textures loaded.
	 */
	void clear();

	/**
	 * Error message from last error or NULL, if no error.
	 */
	const char *TextureManager::getLastErrorString() {
		return lastError;
	}

	// get Singleton instance or create it
	static TextureManager* getSingletonPtr();

protected:
	// Singleton, Ctor + Dtor protected
	TextureManager();
	~TextureManager();

	/// A list of all the textures loaded
	typedef stdext::hash_map<std::string, BitmapTexture*> TextureList;
	TextureList m_Textures;

	// last error message
	char lastError[500];

private:
};


};