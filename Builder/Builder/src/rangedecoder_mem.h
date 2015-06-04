/*
===============================================================================

  FILE:  rangedecoder_mem.h
  
  CONTENTS:
      
  PROGRAMMERS:
  
    Sung-Eui Yoon
  
  COPYRIGHT:
  
    Copyright (C) 2006 Sung-Eui Yoon
    
    This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  CHANGE HISTORY:
  
    27 Sep 2006 -- adopted from Martin's code before leaving Taiwan for PG06
  
===============================================================================
*/
#ifndef RANGEDECODER_MEM_FILE_H
#define RANGEDECODER_MEM_FILE_H

#include <stdio.h>

#include "rangemodel.h"
#include "rangedecoder.h"
#include "memory_map.h"

class RangeDecoderMemFile : public RangeDecoder
{
public:

  CMemoryMappedFile <unsigned char> & m_File;

  RangeDecoderMemFile (CMemoryMappedFile <unsigned char> & File);
  //RangeDecoderMemFile (unsigned char* chars, int number_chars);

  inline unsigned int inbyte();
};


inline RangeDecoderMemFile::RangeDecoderMemFile (CMemoryMappedFile <unsigned char> & File)
: m_File (File)
{
  m_ReadBytes = 0; 
  chars = 0;
  number_chars = 0;
  current_char = 0;

  buffer = inbyte();
  if (buffer != HEADERBYTE)
  {
    fprintf(stderr, "RangeDecoder: wrong HEADERBYTE of %d. is should be %d\n", buffer, HEADERBYTE);
    return;
  }
  buffer = inbyte();
  low = buffer >> (8-EXTRA_BITS);
  range = (unsigned int)1 << EXTRA_BITS;
}

inline unsigned int RangeDecoderMemFile::inbyte (void)
{
  m_ReadBytes++;

  return m_File.GetNextElement ();
  
}
#endif
