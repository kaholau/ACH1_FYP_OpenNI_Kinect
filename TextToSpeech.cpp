/* ----------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** TextToSpeech.cpp
**
** Date: 2016-01-24
** Description: This is the source file of class 'TextToSpeech'. This class
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

#include "TextToSpeech.h"

std::queue<std::wstring> TextToSpeech::strQueue;

TextToSpeech::TextToSpeech()
{
}

TextToSpeech::~TextToSpeech()
{
	if (pVoice != NULL)
	{
		pVoice->Release();
		pVoice = NULL;
	}
	::CoUninitialize();
}

void TextToSpeech::Initialize()
{
	pVoice = NULL;
	if (FAILED(::CoInitialize(NULL)))
		isInitialized = -1;

	isInitialized = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void **)&pVoice);
	if (SUCCEEDED(isInitialized))
		pVoice->SetRate(4);
}

// speak out the first string stored in queue 'strQueue' and pop it
bool TextToSpeech::speak(void)
{
	if (SUCCEEDED(isInitialized))
	{
		if (strQueue.size() > 0) {
			hr = pVoice->Speak(strQueue.front().c_str(), 0, NULL);
			strQueue.pop();

			if (FAILED(hr))
				std::cerr << "Text to speech HR error!\n";
		}
	}
	else
		return false;

	return true;
}

// speak out the text passed immediately
// input datatype: string
bool TextToSpeech::speakNow(std::string &s)
{
	assert(isInitialized);

	std::wstring speech;
	speech = std::wstring(s.begin(), s.end());

	if (SUCCEEDED(hr))
	{
		hr = pVoice->Speak(speech.c_str(), 0, NULL);
	}
	else
		return false;

	return true;
}

// speak out the text passed immediately
// input datatype: wstring
bool TextToSpeech::speakNow(std::wstring &ws)
{
	assert(isInitialized);

	if (SUCCEEDED(hr))
	{
		hr = pVoice->Speak(ws.c_str(), 0, NULL);
	}
	else
		return false;

	return true;
}

// push a text at the end of queue 'strQueue'
// input datatype: string
void TextToSpeech::pushBack(std::string &s)
{
	std::wstring speech;
	speech = std::wstring(s.begin(), s.end());
	strQueue.push(speech);
}

// push a text at the end of queue 'strQueue'
// input datatype: wstring
void TextToSpeech::pushBack(std::wstring &ws)
{
	strQueue.push(ws);
}

void TextToSpeech::testExample(void)
{
	std::string test = "Hello World!";
	this->speakNow(test);
}
