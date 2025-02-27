import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_video_subtitles(url: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Automatically generate video subtitles 
		Automatically generate video subtitles with Speech-to-Text machine learning algorithm inside.
		
		**Supported Video Format**:   .mp4   .mov  .wmv  .avi
		**Output Subtitle Format**:   SRT  ( response with plain text )
		**Average Processing Time**:  3 minutes 
		**Maximum Video Size**:  1GB (1024M) 
		**Supported Languages**:
		    'af': 'Afrikaans',
		    'ar': 'Arabic',
		    'az': 'Azerbaijani',
		    'be': 'Belarusian',
		    'bg': 'Bulgarian',
		    'bn': 'Bengali',
		    'bs': 'Bosnian',
		    'ca': 'Catalan',
		    'ceb': 'Cebuano',
		    'cs': 'Czech',
		    'cy': 'Welsh',
		    'da': 'Danish',
		    'de': 'German',
		    'el': 'Greek',
		    'en': 'English',
		    'eo': 'Esperanto',
		    'es': 'Spanish',
		    'et': 'Estonian',
		    'eu': 'Basque',
		    'fa': 'Persian',
		    'fi': 'Finnish',
		    'fr': 'French',
		    'ga': 'Irish',
		    'gl': 'Galician',
		    'gu': 'Gujarati',
		    'ha': 'Hausa',
		    'hi': 'Hindi',
		    'hmn': 'Hmong',
		    'hr': 'Croatian',
		    'ht': 'Haitian Creole',
		    'hu': 'Hungarian',
		    'hy': 'Armenian',
		    'id': 'Indonesian',
		    'ig': 'Igbo',
		    'is': 'Icelandic',
		    'it': 'Italian',
		    'iw': 'Hebrew',
		    'ja': 'Japanese',
		    'jw': 'Javanese',
		    'ka': 'Georgian',
		    'kk': 'Kazakh',
		    'km': 'Khmer',
		    'kn': 'Kannada',
		    'ko': 'Korean',
		    'la': 'Latin',
		    'lo': 'Lao',
		    'lt': 'Lithuanian',
		    'lv': 'Latvian',
		    'mg': 'Malagasy',
		    'mi': 'Maori',
		    'mk': 'Macedonian',
		    'ml': 'Malayalam',
		    'mn': 'Mongolian',
		    'mr': 'Marathi',
		    'ms': 'Malay',
		    'mt': 'Maltese',
		    'my': 'Myanmar (Burmese)',
		    'ne': 'Nepali',
		    'nl': 'Dutch',
		    'no': 'Norwegian',
		    'ny': 'Chichewa',
		    'pa': 'Punjabi',
		    'pl': 'Polish',
		    'pt': 'Portuguese',
		    'ro': 'Romanian',
		    'ru': 'Russian',
		    'si': 'Sinhala',
		    'sk': 'Slovak',
		    'sl': 'Slovenian',
		    'so': 'Somali',
		    'sq': 'Albanian',
		    'sr': 'Serbian',
		    'st': 'Sesotho',
		    'su': 'Sudanese',
		    'sv': 'Swedish',
		    'sw': 'Swahili',
		    'ta': 'Tamil',
		    'te': 'Telugu',
		    'tg': 'Tajik',
		    'th': 'Thai',
		    'tl': 'Filipino',
		    'tr': 'Turkish',
		    'uk': 'Ukrainian',
		    'ur': 'Urdu',
		    'uz': 'Uzbek',
		    'vi': 'Vietnamese',
		    'yi': 'Yiddish',
		    'yo': 'Yoruba',
		    'zh-CN': 'Chinese (Simplified)',
		    'zh-TW': 'Chinese (Traditional)',
		    'zu': 'Zulu',"
    
    """
    url = f"https://auto-subtitle.p.rapidapi.com/auto_generate_video_subtitle"
    querystring = {'url': url, 'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "auto-subtitle.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

