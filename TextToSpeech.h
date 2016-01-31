/* ----------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** TextToSpeech.h
**
** Date: 2016-01-24
** Description: This is the header file of class 'TextToSpeech'. This class
** requires the use of Microsoft Speech API (SAPI). Please install it and
** include the path and library first.
** 
** Microsoft Speech API (SAPI) 5.3 Reference:
** https://msdn.microsoft.com/en-us/library/ms720161(v=vs.85).aspx
** https://msdn.microsoft.com/en-us/library/ms720163(v=vs.85).aspx
** Microsoft Speech SDK 5.1 Download:
** https://www.microsoft.com/en-us/download/details.aspx?id=10121
**
** 
** Author: Chan Tong Yan, Lau Ka Ho, Joel @ HKUST
** E-mail: yumichanhk2014@gmail.com
**
** --------------------------------------------------------------------------*/
#ifndef _TEXT_TO_SPEECH_H
#define _TEXT_TO_SPEECH_H

#define _ATL_APARTMENT_THREADED

#include <atlbase.h>
extern CComModule _Module;
#include <atlcom.h>

#include <assert.h>
#include <iostream>
#include <sapi.h>
#include <string>
#include <queue>

typedef class TextToSpeech
{
public:
	TextToSpeech();
	~TextToSpeech();


	void Initialize();

	// speak out the first string stored in queue 'strQueue' and pop it
	bool speak(void);

	// speak out the text passed immediately
	// input datatype: string
	bool speakNow(std::string &);

	// speak out the text passed immediately
	// input datatype: wstring
	bool speakNow(std::wstring &);

	// push a text at the end of queue 'strQueue'
	// input datatype: string
	static void pushBack(std::string &);

	// push a text at the end of queue 'strQueue'
	// input datatype: wstring
	static void pushBack(std::wstring &);

	void testExample(void);

private:
	HRESULT isInitialized;
	HRESULT hr;
	ISpVoice *pVoice;
	static std::queue<std::wstring> strQueue;
} TextToSpeech;

#endif // _TEXT_TO_SPEECH_H
