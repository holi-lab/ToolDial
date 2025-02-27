import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def word_list(language: str, category: str, max_list_length: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide a category, a language and optionally a maximum list length. The API will return a list of thematic words in the chosen language."
    language: AF = ' Afrikaans'
AM = ' Amharic'
AR = ' Arabic'
AS = ' Assamese'
AZ = ' Azerbaijani'
BA = ' Bashkir'
BG = ' Bulgarian'
BN = ' Bengali'
BO = ' Tibetan'
BS = 'Bosnian (Latin)'
CA = ' Catalan'
CS = ' Czech'
CY = ' Welsh'
DA = ' Danish'
DE = ' German'
DV = 'Divehi'
EL = ' Greek'
EN = ' English'
ES = ' Spanish'
ET = ' Estonian'
EU = ' Basque'
FA = ' Persian'
FI = ' Finnish'
FIL = 'Filipino'
FJ = ' Fiji'
FO = ' Faeroese'
FR = ' French'
FR_CA = 'French (Canada)'
GA = ' Irish'
GL = ' Galician'
GU = ' Gujarati'
HE = 'Hebrew'
HI = ' Hindi'
HR = ' Croatian'
HSB = 'Upper Sorbian'
HT = ' Haitian Creole'
HU = ' Hungarian'
HY = ' Armenian'
ID = 'Indonesian'
IKT = 'Inuinnaqtun'
IS = ' Icelandic'
IT = ' Italian'
IU = 'Inuktitut'
IU_LATN = 'Inuktitut (Latin)'
JA = ' Japanese'
KA = ' Georgian'
KK = ' Kazakh'
KM = ' Khmer'
KMR = 'Kurdish (Northern)'
KN = ' Kannada'
KO = ' Korean'
KU = ' Kurdish'
KY = ' Kirghiz'
LO = ' Laothian'
LT = ' Lithuanian'
LV = ' Latvian'
LZH = 'Chinese (Literary)'
MG = ' Malagasy'
MI = ' Maori'
MK = ' Macedonian'
ML = ' Malayalam'
MN_CYRL = 'Mongolian (Cyrillic)'
MN_MONG = 'Mongolian (Traditional)'
MR = ' Marathi'
MS = ' Malay'
MT = ' Maltese'
MWW = 'Hmong Daw (Latin)'
MY = ' Burmese'
NB = 'Norwegian'
NE = ' Nepali'
NL = ' Dutch'
OR = ' Oriya'
OTQ = 'Queretaro Otomi'
PA = ' Punjabi'
PL = ' Polish'
PRS = 'Dari'
PS = ' Pashto'
PT = ' Portuguese'
PT_PT = 'Portuguese (Portugal)'
RO = ' Romanian'
RU = ' Russian'
SK = ' Slovak'
SL = ' Slovenian'
SM = ' Samoan'
SO = ' Somali'
SQ = ' Albanian'
SR_CYRL = 'Serbian (Cyrillic)'
SR_LATN = 'Serbian (Latin)'
SV = ' Swedish'
SW = ' Swahili'
TA = ' Tamil'
TE = ' Telugu'
TH = ' Thai'
TI = ' Tigrinya'
TK = ' Turkmen'
TLH_LATN = 'Klingon'
TLH_PIQD = 'Klingon (plqaD)'
TO = ' Tonga'
TR = ' Turkish'
TT = ' Tatar'
TY = 'Tahitian'
UG = ' Uyghur'
UK = ' Ukrainian'
UR = ' Urdu'
UZ = ' Uzbek'
VI = ' Vietnamese'
YUA = 'Yucatec Maya'
YUE = 'Cantonese (Traditional)'
ZH_HANS = 'Chinese Simplified'
ZH_HANT = 'Chinese Traditional'
ZU = ' Zulu'

        
    """
    url = f"https://multi-lingual-word-list.p.rapidapi.com/wordlist/{category}"
    querystring = {'language': language, }
    if max_list_length:
        querystring['max_list_length'] = max_list_length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multi-lingual-word-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of supported languages"
    
    """
    url = f"https://multi-lingual-word-list.p.rapidapi.com/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multi-lingual-word-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

