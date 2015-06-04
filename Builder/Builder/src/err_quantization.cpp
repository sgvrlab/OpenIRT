#include "err_quantization.h"
#include "assert.h"

#include <math.h>

float Log2 (float v)
{
	return (log (v)/log(2.)); 
}

void CErrorQuan::Set (float MinErr, float MaxErr, float ErrRatio, int GivenNumBits)
{
	m_SumQuanErr = 0;
	m_NumFittedData = 0;

	m_MinErr = MinErr;
	m_MaxErr = MaxErr;
	m_ErrRatio = ErrRatio;

	if (GivenNumBits != 0) {
		m_NumUpBit = GivenNumBits;
		// compute simplifiction ratio, which utilize most of the bits.
		m_MaxNumUp = pow ((float)2, (int)GivenNumBits) - 1;
		m_ErrRatio = pow ((m_MaxErr/m_MinErr), 1.f/float(m_MaxNumUp));

		m_ErrRatio *= (1.000001);		// for numerical issue
		assert (m_MinErr*pow (m_ErrRatio, float (m_MaxNumUp)) >= m_MaxErr);
	}
	else {
		// we use geometric distribution on error.
		int temp_MaxNumUp = ceil (log (m_MaxErr/m_MinErr)/log(m_ErrRatio));

		m_NumUpBit = ceil (Log2 (float (temp_MaxNumUp)));
		m_MaxNumUp = pow ((float)2, (int)m_NumUpBit) - 1;		// fully use all the bits
		m_ErrRatio = pow ((m_MaxErr/m_MinErr), 1.f/float(m_MaxNumUp));

		assert (m_MinErr*pow (m_ErrRatio, float (m_MaxNumUp)) >= m_MaxErr);

		
	}			

	/*
	//m_ShiftAmt = ceil (Log2 (ErrRation));	
	m_ShiftAmt = m_NumLowBit = ceil (Log2 (float (ErrRatio)));	// decide upper bit shift amount

	// compute initial error
	m_InitErr = m_MinErr / pow (2, m_ShiftAmt);

	assert (m_InitErr * pow (2, m_ShiftAmt) >= m_MinErr);	// check the minimum case
	

	int i;
	for (i = 1;i < 1000000;i++)
		if (m_InitErr*pow (2, i) > m_MaxErr) {
			m_NumUpBit = ceil (Log2 (float (i)));
			break;
		}

	m_MaxNumUp = pow (2, m_NumUpBit);
	m_MaxNumLow = pow (2, m_NumLowBit);

	// check the maximum case
	assert (m_InitErr * pow (2, m_MaxNumUp*m_ShiftAmt) +
			m_InitErr * pow (2, (m_MaxNumUp - 1)*m_ShiftAmt) * (m_MaxNumLow - 1) >= m_MaxErr);	// check the minimum case
	*/

}

bool CErrorQuan::Fit (float InputErr, unsigned int & UpperCom)
{
	// we use geometric distribution on error.
	// 0.0000001 for numerical stability
	UpperCom = ceil (log (InputErr/m_MinErr)/log(m_ErrRatio) + 0.0001f);

	if (UpperCom >= m_MaxNumUp)		
		UpperCom = m_MaxNumUp;

	float QuantizedV = m_MinErr*pow (m_ErrRatio, float (UpperCom));
	assert (QuantizedV >= InputErr);

	float QuantizationErrorRatio = (QuantizedV - InputErr)/InputErr;
	m_SumQuanErr += QuantizationErrorRatio;
	m_NumFittedData++;


	/*
	UpperCom = 0;		// initializa that it cannot have
	// decide upperCom
	int i;
	float Total, UpperPart, LowPart;
	for (i = 2;i < m_MaxNumUp;i++)
	{
		UpperPart = m_InitErr * pow (2, i*m_ShiftAmt);
		if (UpperPart >= InputErr) {
			UpperCom = i - 1;
			break;
		}
	}

	assert (UpperCom != 0);

	UpperPart = m_InitErr * pow (2, UpperCom*m_ShiftAmt);

	// decide LowerCom
	const int MaxLowerCom = 1024*1024;
	LowerCom = MaxLowerCom;		// initializa that it cannot have
	for (i = 0;i < m_MaxNumLow;i++)
	{
		LowPart = m_InitErr * pow (2, (UpperCom - 1)*m_ShiftAmt)*i;
		Total = UpperPart + LowPart;

		if (LowPart >= InputErr) {
			LowerCom = i - 1;
			break;
		}
	}

	if (LowerCom == MaxLowerCom) {
		// we need to carry on to conservative quantize
		UpperCom++;
		LowerCom = 0;
	}

	
	// error compute
	UpperPart = m_InitErr * pow (2, UpperCom*m_ShiftAmt);
	LowPart = m_InitErr * pow (2, (UpperCom - 1)*m_ShiftAmt)*LowerCom;

	float QuantizedValue = UpperPart + LowPart;		
	assert (QuantizedValue >= InputErr);	// make sure conservativeness

	float QuantizationErrorRatio = (QuantizedValue - InputErr)/InputErr;
	m_SumQuanErr += QuantizationErrorRatio;
	m_NumFittedData++;
	*/

	return true;
}
float CErrorQuan::GetAvgQuantizationErr (void)
{
	return 0.;
}
