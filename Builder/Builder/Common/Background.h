#ifndef COMMON_BACKGROUND_H
#define COMMON_BACKGROUND_H

/********************************************************************
	created:	2004/10/16
	created:	16:10:2004   1:18
	filename: 	c:\MSDev\MyProjects\Renderer\Common\Background.h
	file path:	c:\MSDev\MyProjects\Renderer\Common
	file base:	Background
	file ext:	h
	author:		Christian Lauterbach (lauterb@informatik.uni-bremen.de)
	
	purpose:	Various classes for rendering a background color, for 
	            example by environment mapping
*********************************************************************/


/**
 * Base class for all backgrounds. Backgrounds must implement the method
 * shade() which is supposed to return a resulting background color which
 * is then written directly to the output image. For calculation purposes,
 * the method gets the direction that the background is looked at as a 
 * parameter (e.g. for indexing the background texture)
 */
class Background
{
public:

	virtual void shade(Vector3 &direction, rgb &backgroundColor) = 0;

	virtual ~Background() {};
protected:
	
private:
};

/**
 * Constant-colored background.
 */
class BackgroundConstant : public Background
{
public:
	BackgroundConstant(rgb &bgcolor) {
		color = bgcolor;
	}
	BackgroundConstant() {
		color = rgb(0,0,0);
	}
	~BackgroundConstant() {};

	void shade(Vector3 &direction, rgb &backgroundColor);

protected:
	rgb color;

};

/**
 * Background environment map with the simplest mapping, spherical coordinates.
 * The texture image is a normal undistorted 2D image. If possible, it should be
 * tileable, otherwise there will be seams.
 * 
 * (see Blinn,Newell (1976): "Texture and reflection in computer generated images")
 */
class BackgroundEMSpherical : public Background 
{
public:
	BackgroundEMSpherical(const char *textureFileName) {
		TextureManager *texMan = TextureManager::getSingletonPtr();
		OptionManager *opt = OptionManager::getSingletonPtr();
		const char *backgroundBasePath = opt->getOption("global", "backgroundPath");

		texFileName = _strdup(textureFileName);
		texPtr = texMan->loadTexture(textureFileName, backgroundBasePath);

		if (!texPtr) {
			LogManager *log = LogManager::getSingletonPtr();
			log->logMessage(LOG_ERROR, "Background: could not load texture for environment map:");
			log->logMessage(LOG_ERROR, texMan->getLastErrorString());
		}
	}

	~BackgroundEMSpherical() {
		TextureManager *texMan = TextureManager::getSingletonPtr();
		if (texPtr) {
			texMan->unloadTexture(texFileName);
		}
		free(texFileName);		
	}

	void shade(Vector3 &direction, rgb &backgroundColor);

protected:
	// bitmap we're using as an environment map
	BitmapTexture *texPtr;
	char *texFileName;	
};

/**
* Background environment map with cube mapping.
*/
class BackgroundEMCubeMap : public Background 
{
public:
	BackgroundEMCubeMap(const char *textureFileName) {
		TextureManager *texMan = TextureManager::getSingletonPtr();
		OptionManager *opt = OptionManager::getSingletonPtr();
		const char *backgroundBasePath = opt->getOption("global", "backgroundPath");

		static const char *file_suffixes[] = {"_RT.jpg", "_LF.jpg", "_UP.jpg", "_DN.jpg", "_BK.jpg", "_FR.jpg"};
		//static const char *file_suffixes[] = {"_RT.TGA", "_LF.TGA", "_UP.TGA", "_DN.TGA", "_BK.TGA", "_FR.TGA"};
		char *texFileName;
		texFileName = new char[MAX_PATH];

		for (int i = 0; i < 6; i++) {
			sprintf(texFileName, "%s%s", textureFileName, file_suffixes[i]);				
			texPtr[i] = texMan->loadTexture(texFileName, backgroundBasePath);

			if (!texPtr[i]) {
				LogManager *log = LogManager::getSingletonPtr();
				log->logMessage(LOG_ERROR, "Background: could not load texture for environment map:");
				log->logMessage(LOG_ERROR, texMan->getLastErrorString());
				return;
			}
			
		}	

		delete texFileName;
	}

	~BackgroundEMCubeMap() {
		TextureManager *texMan = TextureManager::getSingletonPtr();

		for (int i = 0; i < 6; i++) 
			if (texPtr[i]) {
				texMan->unloadTexture(texPtr[i]);
			}
			
	}

	void shade(Vector3 &direction, rgb &backgroundColor);

protected:
	// bitmap we're using as an environment map
	BitmapTexture *texPtr[6];	
};


#endif