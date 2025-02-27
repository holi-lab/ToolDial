import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getbooks(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all of the books of the Bible in the language specified with the '?language=[language]' parameter. Currently supported languages are English, Spanish, and Arabic; with more on the way."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetBooks"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparables_new(language: str='english', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Produces a list of parables in the Bible and their relative citations (e.g., 'GetParables?language=english'). The verses of the parables can be retrieved in your app by using the 'GetParseCitation' call (q.v.). For example, for the last parable, 'The Sheep and the Goats', we can retrieve the 'verseIds' for that parable with this call: 'GetParseCitation?citation=Matthew 25:31-46'."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetParables"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchapterbybookandchapterid_new(versionid: str, bookandchapterid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Developed to use with the 'GetBibleReadingPlan', it returns a complete Bible chapter according to the 'bookAndChapterId' and 'versionId' submitted."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetChapterByBookAndChapterId"
    querystring = {'versionId': versionid, 'bookAndChapterId': bookandchapterid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsearchadvanced_new(versionid: str, query: str, excludestring: str='Urias', matchtype: str='exact', limittobookid: str='40', limittochapterid: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Advanced Search is a more powerful search than the 'GetSearch' GET request. For example, 'GetSearchAdvanced?query=David&versionId=kjv&matchType=exact&excludeString=urias&limitToBookId=40&limitToChapterId=1' will search for 'David' in the KJV and exclude any mention of 'Urias' within the book of Matthew (bookId: '40') and within chapter '1' only.
		
		The 'matchType' parameter can be set to 'broad' or 'exact'. A broad setting will include any general matches, for example, 'GetSearchAdvanced?query=da&versionId=kjv&matchType=broad' will return all instances wherein the string, 'da', is found. So, a verse containing, 'darkness' would be considered correct. However, if we limit the search to exact with the 'matchType' parameter, only an exact match would return. For example, 'GetSearchAdvanced?query=da&versionId=kjv&matchType=exact' would give back an empty response."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetSearchAdvanced"
    querystring = {'versionId': versionid, 'query': query, }
    if excludestring:
        querystring['excludeString'] = excludestring
    if matchtype:
        querystring['matchType'] = matchtype
    if limittobookid:
        querystring['limitToBookId'] = limittobookid
    if limittochapterid:
        querystring['limitToChapterId'] = limittochapterid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstrongs(lexiconid: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Strong's in Hebrew or Greek. The [id] in the JSON returned corresponds to the 'H' for Hebrew and 'G' for Greek that Strong's uses to precede the id. Thus, an [id] of 3 in our table is equivalent to Strong's Hebrew H3 as well as Strong's Greek G3. The Strong's id needed can be ascertained after using a GetOriginalText call (e.g. 'GetOriginalText?verseId=01001001') wherein we can receive back the Strong's Ids for any word in Hebrew or Greek respectively."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetStrongs"
    querystring = {'lexiconId': lexiconid, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getreadingtimebyage_new(requestedage: str, wordcount: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This response contains information about how long, on average, it should take a reader of a specified age to read a number of words."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetReadingTimeByAge"
    querystring = {'requestedAge': requestedage, 'wordCount': wordcount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbookandchapternamebybookandchapterid_new(bookandchapterid: str='01001', language: str='english', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the name of the book and chapter as specified in the 'bookAndChapterId' and 'language' parameter values. Both parameters are required."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetBookAndChapterNameByBookAndChapterId"
    querystring = {}
    if bookandchapterid:
        querystring['bookAndChapterId'] = bookandchapterid
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbooknamebyverseid(verseid: int, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the name of the book per the verseId sent in the language specified. For example, 'GetBookNameByVerseId?verseId=40001001&language=english' would return 'Matthew'."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetBookNameByVerseId"
    querystring = {'verseId': verseid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbooknamebybookid(bookid: int, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the name of the book per the number sent in the language specified. For example, 'GetBookNameByBookId?bookId=01&language=english' would return 'Genesis'."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetBookNameByBookId"
    querystring = {'bookId': bookid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparsecitation(citation: str='Matt.1:2,3,7,10-14', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all of the verseIds for the citation submitted. You can use abbreviations (e.g., Ex., or Exod. for Exodus) and multiple references within the same chapter. Also, see GetBibleBookAbbreviations for an array of abbreviations."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetParseCitation"
    querystring = {}
    if citation:
        querystring['citation'] = citation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethebrewcharactersandunicodepoints(language: str='english', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the Unicode for the Hebrew alphabet and corresponding English transliterations (based on the Unicode standard). The response includes letters, points, accents, punctuation, marks, signs, and Yiddish ligatures."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetHebrewCharactersAndUnicodePoints"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getoriginaltext(verseid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return either the Hebrew or the Greek depending on the Testament assessed from the verseId. There are thirty-nine (39) books in the Old Testament (OT)."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetOriginalText"
    querystring = {'verseId': verseid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwordcountofbook_new(bookid: str='01', versionid: str='kjv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a count of all of the words in the specified book."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetWordCountOfBook"
    querystring = {}
    if bookid:
        querystring['bookId'] = bookid
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwordcountofverse_new(verseid: str='01001001', versionid: str='kjv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint's response will contain a count of all of the words of any given verse."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetWordCountOfVerse"
    querystring = {}
    if verseid:
        querystring['verseId'] = verseid
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbooksot(language: str='english', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of only the Old Testament books in the language specified with the '?language=[language]' parameter."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetBooksOT"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbiblebookabbreviations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetBibleBookAbbreviations will return an array of all the abbreviations for bible book names. Also see, GetParseCitation."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetBibleBookAbbreviations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchapter(versionid: str, chapterid: int, bookid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a complete Bible chapter. Required parameters: bookId, chapterId, and versionId. For example, 'GetChapter?bookId=01&chapterId=02&versionId=kjv' would return the entire second chapter of Genesis in the King James Version."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetChapter"
    querystring = {'versionId': versionid, 'chapterId': chapterid, 'bookId': bookid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchaptercount(bookid: str='66', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns simply the number of chapters in any book requested via the 'bookId' parameter. For example, 'GetChapterCount?bookId=66' would return '22' as their are twenty-two (22) chapters in Revelation - the 66th book."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetChapterCount"
    querystring = {}
    if bookid:
        querystring['bookId'] = bookid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcrossreferences(verseid: str='01001001', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all of the complete cross-references (with starting and ending verse, if applicable) for the verseId received."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetCrossReferences"
    querystring = {}
    if verseid:
        querystring['verseId'] = verseid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinfo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetInfo will return information about this API."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetInfo"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparallelverses(verseid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the verse according to the verseId sent in all of the versions available."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetParallelVerses"
    querystring = {'verseId': verseid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrandomchapter(versionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random chapter in its entirety."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetRandomChapter"
    querystring = {'versionId': versionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstories(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return a JSON containing all of the stories by their starting and ending verse of the Bible. Currently, only in English (?language=english), but more are in active development."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetStories"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwords(versionid: str, verseid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return a word-by-word JSON array of the specified verse, along with the word count."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetWords"
    querystring = {'versionId': versionid, 'verseId': verseid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getverse(verseid: str, versionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return a single verse from the Bible using the 'verseId', which is composed of eight (8) digits. The first two (2) are the book number, the second three (3), the chapter number, and the last three (3) are the verse number."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetVerse"
    querystring = {'verseId': verseid, 'versionId': versionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwordsofjesus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return an array of all the verseIds wherein Jesus spoke - just as you would find in a 'red letter edition' bible. This endpoint does not take any parameters."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetWordsOfJesus"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgreekcharactersandunicode(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the Unicode for the Greek alphabet in both lower and uppercase."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetGreekCharactersAndUnicode"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getversions_newly_updated(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return an array of the bible versions available in the API.
		
		Because Bible versions are standardized across the world, we do not attempt to translate versions programmatically.
		
		English versions: ASV (American Standard 1901), BBE (Bible in Basic English), DBY (Darby English Bible), KJV (King James Version), KJV1611 (King James Version 1611), WBT (Webster Bible), WEB (World English Bible), YLT (Young Literal Translation)
		
		Spanish version: RV1909 (Reina-Valera 1909)
		
		Arabic version: SVD (Smith-Van Dyke)"
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetVersions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaudionarration_newly_updated(versionid: str, chapterid: str, bookid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the audio narration file for the Bible chapter in the version specified. Currently supported versions: KJV, RV1909 (Reina Valera 1909 (Spanish)), and SVD (Smith-Van Dyke (Arabic))."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetAudioNarration"
    querystring = {'versionId': versionid, 'chapterId': chapterid, 'bookId': bookid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwordcountofchapter_new(versionid: str='kjv', bookandchapterid: str='01001', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return a count of all of the words in any given chapter."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetWordCountOfChapter"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if bookandchapterid:
        querystring['bookAndChapterId'] = bookandchapterid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparseverseid_new(verseid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parses any 'verseId' to return additional IDs for books, chapters, and verses."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetParseVerseId"
    querystring = {'verseId': verseid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbiblereadingplan_new(days: str, requestedstartdate: str='2023-01-01', requestedage: str='15', sections: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return a Bible reading plan dividing the Bible into chapters according to the days specified. There are optional parameters that you can set after the required 'days' parameter."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetBibleReadingPlan"
    querystring = {'days': days, }
    if requestedstartdate:
        querystring['requestedStartDate'] = requestedstartdate
    if requestedage:
        querystring['requestedAge'] = requestedage
    if sections:
        querystring['sections'] = sections
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrandomverse_newly_updated(versionid: str, limittochapterid: str='1', limittobookid: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random verse from the Bible (Old Testament or New Testament). The 'GetRandomVerse' has been newly updated and now features advanced filter parameters, such as limiting the random verse to the New or Old Testament, limiting the random verse to a specific book; or a specific book and chapter. E.g., 'GetRandomVerse?versionId=kjv&limitToBookId=20&limitToChapterId=1' will limit to the book of Proverbs, chapter '1'. Or, 'GetRandomVerse?versionId=kjv&limitToBookId=20' will limit to the book of Proverbs, any chapter. If no parameters are added after 'versionId', a random verse from the entire Bible will be requested."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetRandomVerse"
    querystring = {'versionId': versionid, }
    if limittochapterid:
        querystring['limitToChapterId'] = limittochapterid
    if limittobookid:
        querystring['limitToBookId'] = limittobookid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcommentary(verseid: str, commentaryname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a commentary in plain text for the commentary name and verse specified. Required parameters: commentaryName, and verseId. Note: Currently, Gill's Commentary is available with more scheduled to be added in the future."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetCommentary"
    querystring = {'verseId': verseid, 'commentaryName': commentaryname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdefinitionbiblical(dictionaryid: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the biblical definition of a query. Currently, Smith's Bible Dictionary is available with more scheduled to be added in the future."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetDefinitionBiblical"
    querystring = {'dictionaryId': dictionaryid, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getversecount(bookid: str, chapterid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the number of verses for the specified book and chapter."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetVerseCount"
    querystring = {'bookId': bookid, 'chapterId': chapterid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbookidbybookname(bookname: str='John', language: str='english', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the bookId of the bookName submitted. At this time, abbreviations, such as with the GetParseCitation endpoint are not supported, but will be in the future."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetBookIdByBookName"
    querystring = {}
    if bookname:
        querystring['bookName'] = bookname
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpropheciesfulfilledinjesus(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the prophecies fulfilled in Jesus."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetPropheciesFulfilledInJesus"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsearchcount(query: str, versionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetSearchCount will return the total count of results for any search term (query) entered. For example, 'GetSearchCount?query=Jesus&versionId=kjv' will return 943 - the number of times that the word, "Jesus" appears in the KJV."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetSearchCount"
    querystring = {'query': query, 'versionId': versionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsearch(versionid: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return all the results of the query entered. For example, 'GetSearch?query=Jesus&versionId=kjv ' will return all the results of the Bible wherein 'Jesus' is found."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetSearch"
    querystring = {'versionId': versionid, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbooksnt(language: str='english', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of only the New Testament books in the language specified with the '?language=[language]' parameter."
    
    """
    url = f"https://iq-bible.p.rapidapi.com/GetBooksNT"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iq-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

