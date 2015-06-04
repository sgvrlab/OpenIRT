
// SampleMFCDlg.cpp : implementation file
//

#include "stdafx.h"
#include "SampleMFC.h"
#include "SampleMFCDlg.h"
#include "afxdialogex.h"

#include "OpenIRT.h"
#include "Image.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CSampleMFCDlg dialog




CSampleMFCDlg::CSampleMFCDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CSampleMFCDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CSampleMFCDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CSampleMFCDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_DESTROY()
END_MESSAGE_MAP()


// CSampleMFCDlg message handlers

BOOL CSampleMFCDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	m_width = 512;
	m_height = 512;

	CWnd *imagePlane = GetDlgItem(IDC_STATIC_IMAGE);

	imagePlane->SetWindowPos(NULL, 0, 0, m_width, m_height, SWP_NOMOVE);

	CRect rectWindow, rectClient, rectFrame, rectMargin, rectImage;

	GetWindowRect(&rectWindow);
	GetClientRect(&rectClient);
	imagePlane->GetWindowRect(&rectImage);

	ClientToScreen(&rectClient);

	rectFrame.left = rectClient.left - rectWindow.left;
	rectFrame.top = rectClient.top - rectWindow.top;
	rectFrame.right = rectWindow.right - rectClient.right;
	rectFrame.bottom = rectWindow.bottom - rectClient.bottom;

	rectMargin.left = rectMargin.right = rectMargin.top = rectMargin.bottom = rectImage.left - rectClient.left;

	SetWindowPos(NULL, 0, 0, 
		m_width + rectFrame.left + rectFrame.right + rectMargin.left + rectMargin.right,
		m_height + rectFrame.top + rectFrame.bottom + rectMargin.top + rectMargin.bottom,
		SWP_NOMOVE);
	m_GLDevice.create(GetDlgItem(IDC_STATIC_IMAGE)->GetDC()->m_hDC);

	OpenIRT *renderer = OpenIRT::getSingletonPtr();

	renderer->pushCamera("Camera1", 220.0f, 380.0f, -10.0f, 0.0f, 380.0f, -10.0f, 0.0f, 1.0f, 0.0f, 72.0f, 1.0f, 1.0f, 100000.0f);

	renderer->loadScene("..\\media\\sponza.scene");

	renderer->init(RendererType::CUDA_PATH_TRACER, m_width, m_height, m_GLDevice.getRenderContext(), m_GLDevice.getDeviceContext());
	m_bindDevice = false;

	Controller &control = *renderer->getController();
	control.drawBackground = true;

	return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CSampleMFCDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}

	m_GLDevice.makeCurrent();

	OpenIRT *renderer = OpenIRT::getSingletonPtr();

	if(m_bindDevice)
	{
		renderer->render(NULL);

		SwapBuffers(m_GLDevice.getDeviceContext());
	}
	else
	{
		static CDC bmpDC;
		static CBitmap bmp;
		static bool isFirst = true;
		static irt::Image image(m_width, m_height, 4); 
		static irt::Image imageOut(m_width, m_height, 4); 

		CDC *pDC = GetDlgItem(IDC_STATIC_IMAGE)->GetDC();
		if(isFirst)
		{
			bmpDC.CreateCompatibleDC(pDC);
			bmp.CreateCompatibleBitmap(pDC, m_width, m_height);
			bmpDC.SelectObject(&bmp);
		}

		renderer->render(&image);

		for(int i=0;i<m_height;i++)
		{
			for(int j=0;j<m_width;j++)
			{
				int offset = (i*m_width + j)*image.bpp;
				int offset2 = ((m_height-i-1)*m_width + j)*image.bpp;

				imageOut.data[offset + 0] = image.data[offset2 + 2];
				imageOut.data[offset + 1] = image.data[offset2 + 1];
				imageOut.data[offset + 2] = image.data[offset2 + 0];
				imageOut.data[offset + 3] = image.data[offset2 + 3];
			}
		}

		bmp.SetBitmapBits(m_width*m_height*4, imageOut.data);

		pDC->BitBlt(0, 0, m_width, m_height, &bmpDC, 0, 0, SRCCOPY);
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CSampleMFCDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CSampleMFCDlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	OpenIRT::getSingletonPtr()->doneRenderer();
}
