/*
===============================================================================

  FILE:  rangedecoder_file.h
  
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
#ifndef RANGEDECODER_FILE_H
#define RANGEDECODER_FILE_H

#include <stdio.h>

#include "rangemodel.h"
#include "rangedecoder.h"

class RangeDecoderFile : public RangeDecoder
{
public:

  FILE * m_pFile;
  RangeDecoderFile (FILE* fp);
  RangeDecoderFile (unsigned char* chars, int number_chars);

  inline unsigned int inbyte();
};

inline RangeDecoderFile::RangeDecoderFile (unsigned char* chars, int number_chars)
{
  m_ReadBytes = 0;

  this->chars = chars;
  this->number_chars = number_chars;
  current_char = 0;
  //fp = 0;

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

inline RangeDecoderFile::RangeDecoderFile (FILE * fp)
{
  m_pFile = fp;

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

inline unsigned int RangeDecoderFile::inbyte (void)
{
  int c;
  if (m_pFile)
  {
    c = getc(m_pFile);
  }
  else
  {
    if (current_char < number_chars)
    {
      c = chars[current_char++];
    }
    else
    {
      c = EOF;
    }
  }

  m_ReadBytes++;

  return c;

}
#endif
