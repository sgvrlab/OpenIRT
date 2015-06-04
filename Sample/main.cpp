#include "OpenIRT.h"
#include "ImageIL.h"

void main(void)
{
	int width = 512, height = 512;

	OpenIRT *renderer = OpenIRT::getSingletonPtr();
	renderer->pushCamera("Camera1", 
			220.0f, 380.0f, -10.0f, 
			0.0f, 380.0f, -10.0f,
			0.0f, 1.0f, 0.0f,
			72.0f, 1.0f, 1.0f, 100000.0f);
	renderer->loadScene("..\\media\\sponza.scene");
	renderer->init(RendererType::CUDA_PATH_TRACER, width, height);
	Controller &control = *renderer->getController();
	control.drawBackground = true;
	irt::ImageIL img(width, height, 4);
	renderer->render(&img);
	img.writeToFile("result.png");
	renderer->doneRenderer();
}
