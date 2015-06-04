
// SampleMFCDlg.h : header file
//

#pragma once

#include "OpenGLDevice.h"

// CSampleMFCDlg dialog
class CSampleMFCDlg : public CDialogEx
{
// Construction
public:
	CSampleMFCDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_SAMPLEMFC_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	OpenGLDevice m_GLDevice;
	bool m_bindDevice;

	int m_width;
	int m_height;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnDestroy();
};
