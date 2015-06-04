#ifndef BITCOMPRESSION_HPP
#define BITCOMPRESSION_HPP
//#include "memory_map.h"
#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#include <stdio.h>
#include <stdlib.h>

class BitCompression
{
public :
	/*
	BitCompression(CMemoryMappedFile <unsigned char> & File)
		: m_File(File)
	{
		chars = 0;
		number_chars = 0;
		current_char = 0;

		buffer = inbyte();
		if (buffer != HEADERBYTE)
		{
			printf ("Error happens here\n");

			fprintf(stderr, "RangeDecoder_MEM: wrong HEADERBYTE of %d. is should be %d\n", buffer, HEADERBYTE);
			return;
		}
		buffer = inbyte();

		localOffset = 0;
		buffer = 0;
	}
	*/

	BitCompression(FILE *fp, int readWrite)
		: fp(fp), readWrite(readWrite)
	{
		if(fp)
		{
			/*
			unsigned char data;
			readWrite = fread(&data, 1, 1, fp) != 0;
			if(readWrite)
				fseek(fp, -1L, SEEK_CUR);
			*/
		}
		if(readWrite)
		{
			chars = 0;
			number_chars = 0;
			current_char = 0;

			localOffset = 0;
			//buffer = inbyte();
		}
		else
		{
			number_chars = 0;
			current_char = 0;
			localOffset = 0;
			buffer = 0;
			chars = 0;

			if(!fp)
			{
				chars = (unsigned char*)malloc(sizeof(unsigned char)*1000);
				allocated_chars = 1000;
			}
		}
	}

	BitCompression(unsigned char* chars, int max_number_chars, int readWrite)
		: chars(chars), max_number_chars(max_number_chars), readWrite(readWrite)
	{
		number_chars = 0;
		current_char = 0;
		fp = 0;

		localOffset = 0;
		//buffer = inbyte();
	}

	BitCompression()
	{
	}

	~BitCompression()
	{
		/*
		if(readWrite)
		{
		}
		else
		{
			buffer <<= (8-localOffset);
			outbyte(buffer);
		}
		*/
		//if(fp)
		//	fclose(fp);
		//fp = 0;
		if(readWrite)
		{
		}
		else
		{
			if(chars)
			{
				free(chars);
			}
		}
	}

	virtual void done() 
	{
		if(readWrite)
		{
			/*
			if(fp && localOffset == 0)
			{
				fseek(fp, -1, SEEK_CUR);
			}
			*/
		}
		else
		{
			if(localOffset)
			{
				buffer <<= (8-localOffset);
				outbyte(buffer);
				localOffset = 0;
			}
		}
	}

	unsigned int buffer;
	int localOffset;
	int readWrite;

	FILE *fp;
	/*
	CMemoryMappedFile <unsigned char> & m_File;
	*/

	unsigned char* chars;
	int current_char;
	int number_chars;
	int max_number_chars;
	int allocated_chars;

	unsigned char* getChars() {return chars;}

	void encode(int nbits, unsigned int data)
	{
		//printf("Encode [%d] : %d\n", nbits, data);
		int remainBits = nbits;
		int currentBits;
		while(remainBits > 0)
		{
			currentBits = min(8-localOffset, remainBits);
			buffer = (buffer << currentBits) | (data >> (remainBits-currentBits));
			localOffset += currentBits;
			if(localOffset == 8) {localOffset = 0; outbyte(buffer); buffer = 0;}
			remainBits -= currentBits;
		}
	}

	void encodeInt(unsigned int data)
	{
		encode(32, data);
	}

	void encodeFloat(float data)
	{
		encodeInt(*((unsigned int*)(&data)));
	}

	unsigned int decode(int nbits)
	{
		int remainBits = nbits;
		unsigned int data = 0;
		int currentBits;
		while(remainBits > 0)
		{
			if(localOffset == 0) buffer = inbyte();
			currentBits = min(8-localOffset, remainBits);
			//if(localOffset == 0) buffer = inbyte();
			//tmp = ;
			data = (data << currentBits) | (((buffer << localOffset) & 0xFF) >> (8-currentBits));
			localOffset += currentBits;
			localOffset = localOffset == 8 ? 0 : localOffset;
			//if(localOffset == 8) {localOffset = 0; buffer = inbyte();}
			remainBits -= currentBits;
		}
		//printf("Decode [%d] : %d\n", nbits, data);
		return data;
	}

	unsigned int decodeInt()
	{
		return decode(32);
	}

	float decodeFloat()
	{
		float dataFloat;
		*((unsigned int*)(&dataFloat)) = decodeInt();
		return dataFloat;
	}

	virtual inline unsigned int inbyte();
	inline void outbyte(unsigned int data);
	//inline unsigned int inbyteMM();

	unsigned int getNumberChars() {return number_chars;}
	unsigned int getNumberBits() {return 8*number_chars;}
	unsigned int getNumberBytes() {return number_chars;}
	unsigned int GetReadBytes() {return number_chars;}

};
inline unsigned int BitCompression::inbyte()
{
	int c;
	if (fp)
	{
		c = getc(fp);
		//printf("Get %d\n", c & 0xFF);
	}
	else
	{
		if (current_char < max_number_chars)
		{
			c = chars[current_char++];
		}
		else
		{
			c = EOF;
		}
	}
	number_chars++;
	return c;
}

inline void BitCompression::outbyte(unsigned int data)
{
	unsigned char c = data & 0xFF;
	if(fp)
	{
		putc(c, fp);
		//printf("Put %d\n", data & 0xFF);
	}
	else
	{
		if(chars)
		{
			if(number_chars == allocated_chars)
			{
				unsigned char* newchars = (unsigned char*) malloc(sizeof(unsigned char)*allocated_chars*2);
				memcpy(newchars,chars,sizeof(unsigned char)*allocated_chars);
				free(chars);
				chars = newchars;
				allocated_chars = allocated_chars*2;
			}
			chars[number_chars] = c;
		}
	}
	number_chars++;
}
/*
inline unsigned int BitCompression::inbyteMM()
{
	return m_File.GetNextElement ();
}
*/
#endif