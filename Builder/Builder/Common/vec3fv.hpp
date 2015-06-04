//============================================================================
// vec3fv.hpp
//============================================================================

void Set3fv(float v[3], float x, float y, float z);
void Copy3fv(float A[3], const float B[3]); // A=B

void ScalarMult3fv(float c[3], const float a[3], float s);
void ScalarDiv3fv(float v[3], float s);
void Add3fv(float c[3], const float a[3], const float b[3]); // c = a + b
void Subtract3fv(float c[3], const float a[3], const float b[3]); // c = a - b
void Negate3fv(float a[3], const float b[3]);  // a = -b

float Length3fv(const float v[3]);
void Normalize3fv(float v[3]);
float DotProd3fv(const float a[3], const float b[3]);
void CrossProd3fv(float* C, const float* A, const float* B); // C = A X B
