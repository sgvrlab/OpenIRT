#include "CImg.h"

#include <math.h>
#include <assert.h>
#include <time.h>
#include <string>

#include <vector>
#include <iostream>

#include "math_tools.h"

#include "point.h"

//Enable / Disable debug statements
#define VERBOSE false

//Maximum of attempts of the algorithm for compact mode (5 in the article)
#define MAX_ESSAIS_COMPACT 5

//Enable binary research (to find minimal size)
//#define COMPACT


using namespace std;
using namespace cimg_library;

class PSH {
public:
    CImg<unsigned char> image;

    PSH(CImg<unsigned char> image);

    int perform();
    void display();

    void save(string prefix);
    void save(const char* hash_fname, const char* offsets_fname);


private:

    int perfect_hashing(CImg<unsigned char> &hash, CImg<unsigned char> &offsets, int &n, int &m, int &u, int &r);
    inline bool non_white_pixel(CImg<unsigned char> &img, int i, int j);
    CImg<unsigned char> hash;
    CImg<unsigned char> offsets;




};
