//-----------------------------------------------------------------------------
// @ binstream.h
// ---------------------------------------------------------------------------
// Defines readers and writers for binary streaming of objects.
//-----------------------------------------------------------------------------


#ifndef __BINSTREAM__
#define __BINSTREAM__


//-----------------------------------------------------------------------------
//-- Includes ----------------------------------------------------------------
//-----------------------------------------------------------------------------

#include "glvu_common.h"
#include "vec3f.hpp"
#include "vec2f.hpp"
#include "vec4f.hpp"
//-----------------------------------------------------------------------------
//-- Defines, Constants ------------------------------------------------------
//-----------------------------------------------------------------------------

// Define LITTLE_ENDIAN and BIG_ENDIAN
#ifdef LITTLE_ENDIAN
  #undef LITTLE_ENDIAN
#endif
#ifdef BIG_ENDIAN
  #undef BIG_ENDIAN
#endif
enum { LITTLE_ENDIAN, BIG_ENDIAN };

/*
#ifdef WIN32
  const int IOS_BINARY = ios::binary;
#else
  const int IOS_BINARY = 0;
#endif
*/

// from CARL's code
// Define IOS_BINARY constant
#ifdef PC
#define IOS_BINARY  ios::binary
#else
#define IOS_BINARY  ios::binary
#endif

#if defined(_WIN32) || defined(__INTEL_COMPILER)
#define TEMPL_FRIEND(x) x
#else
#define TEMPL_FRIEND(x) x<>
#endif

//-----------------------------------------------------------------------------
//-- Function Declarations, Macros -------------------------------------------
//-----------------------------------------------------------------------------

// Return the endianness of the machine, either LITTLE_ENDIAN or BIG_ENDIAN
inline int Endian();
// Converts buffer of bytes from little endian to big endian and vice versa
inline void SwapBytes( char* bytes, int numBytes );

/*
// Reads from binary input stream
inline istream& Read( istream& in, bool& b );
inline istream& Read( istream& in, char& c );
inline istream& Read( istream& in, unsigned char& uc );
inline istream& Read( istream& in, signed char& sc );
inline istream& Read( istream& in, unsigned int& ui );
inline istream& Read( istream& in, signed int& si );
inline istream& Read( istream& in, unsigned short int& us );
inline istream& Read( istream& in, signed short int& ss );
inline istream& Read( istream& in, unsigned long int& ul );
inline istream& Read( istream& in, signed long int& sl );
inline istream& Read( istream& in, float& f );
inline istream& Read( istream& in, double& d );
inline istream& Read( istream& in, long double& ld );
// Read pointers
template<class Type> inline istream& Read( istream& in, Type*& pointer );
inline istream& Read( istream& in, void*& v );
inline istream& Read( istream& in, char* c, int maxLen);
// Read vectors
inline istream& Read(istream& in, Vec2f& v);
inline istream& Read(istream& in, Vec3f& v);
inline istream& Read(istream& in, Vec4f& v);

// Writes to binary output stream
inline ostream& Write( ostream& out, bool b );
inline ostream& Write( ostream& out, char c );
inline ostream& Write( ostream& out, unsigned char uc );
inline ostream& Write( ostream& out, signed char sc );
inline ostream& Write( ostream& out, unsigned int ui );
inline ostream& Write( ostream& out, signed int si );
inline ostream& Write( ostream& out, unsigned short int us );
inline ostream& Write( ostream& out, signed short int ss );
inline ostream& Write( ostream& out, unsigned long int ul );
inline ostream& Write( ostream& out, signed long int sl );
inline ostream& Write( ostream& out, float f );
inline ostream& Write( ostream& out, double d );
inline ostream& Write( ostream& out, long double ld );
// Write pointers
template<class Type> inline ostream& Write( ostream& out, Type* pointer );
inline ostream& Write( ostream& out, void* v );
inline ostream& Write( ostream& out, char* c );
// Write vectors
inline ostream& Write(ostream& out, const Vec2f& v);
inline ostream& Write(ostream& out, const Vec3f& v);
inline ostream& Write(ostream& out, const Vec4f& v);
*/



//-----------------------------------------------------------------------------
//-- Typedefs, Structs, Classes ----------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//-- External Variables ------------------------------------------------------
//-----------------------------------------------------------------------------

extern bool switchEndian;

//-----------------------------------------------------------------------------
//-- Function Definitions ----------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read bool
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, bool& b)
{
  unsigned char uc;
  in.read((char*)( &uc ), sizeof( uc ));
	b = uc ? true : false;
  return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read char
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, char& c)
{
  return in.read( &c, sizeof( c ) );
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read unsigned char
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, unsigned char& uc)
{
  return in.read( (char*)( &uc ), sizeof( uc ) );
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read signed char
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, signed char& sc)
{
  return in.read( (char*)( &sc ), sizeof( sc ) );
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read unsigned int
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, unsigned int& ui)
{
  in.read( (char*)( &ui ), sizeof( ui ) );
	if (switchEndian) {
		SwapBytes( (char*)( &ui ), sizeof( ui ) );
	}
	return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read signed int
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, signed int& si)
{
  in.read( (char*)( &si ), sizeof( si ) );
	if (switchEndian) {
		SwapBytes( (char*)( &si ), sizeof( si ) );
	}
	return in;
}  

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read unsigned short int
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, unsigned short int& us)
{
  in.read( (char*)( &us ), sizeof( us ) );
	if (switchEndian) {
		SwapBytes( (char*)( &us ), sizeof( us ) );
	}
	return in;
}  // end of Read()

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read signed short int
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, signed short int& ss)
{
  in.read( (char*)( &ss ), sizeof( ss ) );
	if (switchEndian) {
		SwapBytes( (char*)( &ss ), sizeof( ss ) );
	}
	return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read unsigned long int
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, unsigned long int& ul)
{
  in.read( (char*)( &ul ), sizeof( ul ) );
	if (switchEndian) {
		SwapBytes( (char*)( &ul ), sizeof( ul ) );
	}
	return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read signed long int
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, signed long int& sl)
{
  in.read( (char*)( &sl ), sizeof( sl ) );
	if (switchEndian) {
		SwapBytes( (char*)( &sl ), sizeof( sl ) );
	}
	return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read float
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, float& f)
{
   in.read( (char*)( &f ), sizeof( f ) );
	if (switchEndian) {
		SwapBytes( (char*)( &f ), sizeof( f ) );
	}
	return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read double
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, double& d)
{
	// Hack this so that you only read and write floats, but do math with
	// doubles
	float f;
    in.read( (char*)( &f ), sizeof( f ) );
	if (switchEndian) {
		SwapBytes( (char*)( &f ), sizeof( f ) );
	}
	d = double( f );
	return in;
#if 0
  in.read( (char*)( &d ), sizeof( d ) );
	if (switchEndian) {
		SwapBytes( (char*)( &d ), sizeof( d ) );
	}
	return in;
#endif
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read long double
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, long double& ld)
{
  in.read( (char*)( &ld ), sizeof( ld ) );
	if (switchEndian) {
		SwapBytes( (char*)( &ld ), sizeof( ld ) );
	}
	return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read pointer
//-----------------------------------------------------------------------------
template<class Type> inline istream& Read(istream& in, Type*& pointer)
{
  in.read( (char*)( &pointer ), sizeof( pointer ) );
	if (switchEndian) {
		SwapBytes( (char*)( &pointer ), sizeof( pointer ) );
	}
	return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read void pointer
//-----------------------------------------------------------------------------
inline istream& Read(istream& in, void*& pointer)
{
  in.read( (char*)( &pointer ), sizeof( pointer ) );
	if (switchEndian) {
		SwapBytes( (char*)( &pointer ), sizeof( pointer ) );
	}
	return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read string
//-----------------------------------------------------------------------------
inline istream& Read( istream& in, char* c, int maxLen)
{
  char *t = c-1;
  bool bufFull=false;
  char trash;
  do {
    if (!bufFull) {
      t++;
      in.read(t, sizeof(char));
      if (t-c+1 == maxLen) {
        *t = 0;
        bufFull=true;
      }
    } else {
      in.read(&trash, sizeof(char));
    }
  } while (*t != 0);
  return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read 2D vector
//-----------------------------------------------------------------------------
template <class T>
inline istream& Read(istream& in, Vec2<T>& v)
{
  Read(in, v.x);
  Read(in, v.y);
  return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read 3D vector
//-----------------------------------------------------------------------------
template <class T>
inline istream& Read(istream& in, Vec3<T>& v)
{
  Read(in, v.x);
  Read(in, v.y);
  Read(in, v.z);
  return in;
}

//-----------------------------------------------------------------------------
// @ Read()
// ---------------------------------------------------------------------------
// Read 4D vector
//-----------------------------------------------------------------------------
template <class T>
inline istream& Read(istream& in, Vec4<T>& v)
{
  Read(in, v.x);
  Read(in, v.y);
  Read(in, v.z);
  Read(in, v.w);
  return in;
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write bool
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, bool b)
{
  unsigned char uc = b ? 1 : 0;
  return out.write( (const char*)( &uc ), sizeof( uc ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write char
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, char c)
{
  return out.write( (const char*)( &c ), sizeof( c ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write unsigned char
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, unsigned char uc)
{
  return out.write( (const char*)( &uc ), sizeof( uc ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write signed char
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, signed char sc)
{
  return out.write( (const char*)( &sc ), sizeof( sc ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write unsigned int
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, unsigned int ui)
{
	if (switchEndian) {
		SwapBytes( (char*)( &ui ), sizeof( ui ) );
	}
    return out.write( (const char*)( &ui ), sizeof( ui ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write signed int
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, signed int si)
{
	if (switchEndian) {
		SwapBytes( (char*)( &si ), sizeof( si ) );
	}
    return out.write( (const char*)( &si ), sizeof( si ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write unsigned short int
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, unsigned short int us)
{
	if (switchEndian) {
		SwapBytes( (char*)( &us ), sizeof( us ) );
	}
    return out.write( (const char*)( &us ), sizeof( us ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write signed short int
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, signed short int ss)
{
	if (switchEndian) {
		SwapBytes( (char*)( &ss ), sizeof( ss ) );
	}
    return out.write( (const char*)( &ss ), sizeof( ss ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write unsigned long int
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, unsigned long int ul)
{
	if (switchEndian) {
		SwapBytes( (char*)( &ul ), sizeof( ul ) );
	}
    return out.write( (const char*)( &ul ), sizeof( ul ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write signed long int
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, signed long int sl)
{
	if (switchEndian) {
		SwapBytes( (char*)( &sl ), sizeof( sl ) );
	}
    return out.write( (const char*)( &sl ), sizeof( sl ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write float
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, float f)
{
	if (switchEndian) {
		SwapBytes( (char*)( &f ), sizeof( f ) );
	}
    return out.write( (const char*)( &f ), sizeof( f ) );
}


//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write double
//-----------------------------------------------------------------------------
// changed to meet with Carl's code
inline ostream& Write(ostream& out, double d)
{
	// Hack this so that you only read and write floats, but do math with
	// doubles
	float f = float( d );
	if (switchEndian) {
		SwapBytes( (char*)( &f ), sizeof( f ) );
	}
  return out.write( (const char*)( &f ), sizeof( f ) );
#if 0
	if (switchEndian) {
		SwapBytes( (char*)( &d ), sizeof( d ) );
	}
  return out.write( (const char*)( &d ), sizeof( d ) );
#endif 
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write long double
//-----------------------------------------------------------------------------
inline ostream& Write(ostream& out, long double ld)
{
	if (switchEndian) {
		SwapBytes( (char*)( &ld ), sizeof( ld ) );
	}
  return out.write( (const char*)( &ld ), sizeof( ld ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write pointer
//-----------------------------------------------------------------------------
template<class Type> inline ostream& Write(ostream& out, Type* pointer)
{
	if (switchEndian) {
		SwapBytes( (char*)( &pointer ), sizeof( pointer ) );
	}
    return out.write( (const char*)( &pointer ), sizeof( pointer ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write void pointer
//-----------------------------------------------------------------------------
inline ostream& Write( ostream& out, void* v )
{
	if (switchEndian) {
		SwapBytes( (char*)( &v ), sizeof( v ) );
	}
  return out.write( (const char*)( &v ), sizeof( v ) );
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write string.
//-----------------------------------------------------------------------------
inline ostream& Write( ostream& out, char* c )
{
  return out.write(c, strlen(c)+1);
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write 2D vector
//-----------------------------------------------------------------------------
template <class T>
inline ostream& Write(ostream& out, const Vec2<T>& v)
{
  Write(out,v.x);
  Write(out,v.y);
  return out;
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write 3D vector
//-----------------------------------------------------------------------------
template <class T>
inline ostream& Write(ostream& out, const Vec3<T>& v)
{
  Write(out,v.x);
  Write(out,v.y);
  Write(out,v.z);
  return out;
}

//-----------------------------------------------------------------------------
// @ Write()
// ---------------------------------------------------------------------------
// Write 4D vector
//-----------------------------------------------------------------------------
template <class T>
inline ostream& Write(ostream& out, const Vec4<T>& v)
{
  Write(out,v.x);
  Write(out,v.y);
  Write(out,v.z);
  Write(out,v.w);
  return out;
}

//-----------------------------------------------------------------------------
// @ SwapBytes()
// ---------------------------------------------------------------------------
// Converts buffer of bytes from little endian to big endian and vice versa.
// Code from Greg Turk.
//-----------------------------------------------------------------------------
inline
void
SwapBytes( char* bytes, int numBytes )
{
    int i;

    for (i = 0; i < numBytes / 2; i++) {
		Swap( bytes[ i ], bytes[ (numBytes - 1) - i ] );
    }
}

//-----------------------------------------------------------------------------
// @ Endian()
// ---------------------------------------------------------------------------
// Return the endianness of the machine, either LITTLE_ENDIAN or BIG_ENDIAN.
// Code from Greg Turk.
//-----------------------------------------------------------------------------
inline int Endian()
{
  int i;
  char* c = (char*)( &i );

  i = 0;
  i = 1;

  if (1 == c[ 0 ]) {
  	return LITTLE_ENDIAN;
  }

  if (1 == c[ sizeof( int ) - 1 ]) {
	  return BIG_ENDIAN;
  }
  // Function should never get here
  ASSERT( false , "Should not get here!");
  return LITTLE_ENDIAN;
}


#endif 
