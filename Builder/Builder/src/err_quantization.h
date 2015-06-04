#ifndef ERROR_QUAN_H
#define ERROR_QUAN_H


class CErrorQuan {
public:

	float m_MinErr, m_MaxErr;	// min and max of error in the tree
	float m_ErrRatio;			// the error ratio

	// qunatization equation
	// Err = Initial_E << (UpperCom*m_ErrRatio) + (Initial_E << (UpperCom - 1)m_ErrRatio)*LowerCom;
	// Note:
	//		1. to avoid any exception UpperCom should be bigger than 1.
	//		   so, we set m_MinErr = m_InitialE << 1 where UpperCom is 1
	//		2. In the equation, m_ErrRatio is the m_ShiftAmt;

	//float m_InitErr;
	//unsigned int m_ShiftAmt;					// amount of upper bit shift
	unsigned int m_NumUpBit, m_NumLowBit;		// number of bit
	unsigned int m_MaxNumUp, m_MaxNumLow;		// maximum index given bit

	// stataistics
	float m_SumQuanErr;
	int m_NumFittedData; 

	void Set (float MinErr, float MaxErr, float ErrRatio, int GivenNumBits = 0);
	bool Fit (float InputErr, unsigned int & UpperCom);
	float GetAvgQuantizationErr (void);
};


#endif
