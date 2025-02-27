import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def synonyms(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get synonyms of a word"
    word: Try "lovely".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/synonyms"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def word(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve information about a word. Results can include definitions, part of speech, synonyms, related words, syllables, and pronunciation. This method is useful to see which relationships are attached to which definition and part of speech of a word."
    word: Try "example".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def definitions(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get definitions of a word, including the part of speech."
    word: Try "incredible".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/definitions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def examples(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get examples of how the word is used."
    word: Try "uneventful".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/examples"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rhymes(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of words that rhyme with the given word."
    
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/rhymes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_a_type_of(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finds word that are more general than the given word."
    word: Try "hatchback".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/typeOf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def antonyms(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get antonyms (opposites) of a word."
    word: Try "free".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/antonyms"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def has_types(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more specific examples of types of this word."
    word: Try "purple".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/hasTypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def part_of(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The larger whole to which the word belongs."
    word: Try "finger".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/partOf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def has_parts(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words that are parts of the original word."
    word: Try "building".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/hasParts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_an_instance_of(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words that the parameter word is an example of."
    word: Try "einstein".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/instanceOf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def has_instances(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words that are examples of the parameter word."
    word: Try "president".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/hasInstances"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similar_to(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words that similar to the original word, but are not synonyms."
    word: Try "bloody".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/similarTo"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def also(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Phrases of which the word is a part."
    word: Try "bump".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/also"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def entails(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words that are implied by the original word. Usually used for verbs."
    word: Try "rub".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/entails"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def member_of(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A group to which the word belongs."
    word: Try "dory".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/memberOf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def has_members(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words that belong to the group defined by the parameter word."
    word: Try "cosmos".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/hasMembers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def substance_of(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Substances to which the original word is a part of."
    
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/substanceOf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def has_substances(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words that are substances of the parameter word."
    word: Try "wood".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/hasSubstances"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def in_category(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The domain category to which the original word belongs."
    word: Try "chaotic".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/inCategory"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def has_categories(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Categories of the parameter word."
    word: Try "math".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/hasCategories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usage_of(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words that the original word is a domain usage of."
    word: Try "advil".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/usageOf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def has_usages(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words that are examples of the domain the original word defines."
    word: Try "colloquialism".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/hasUsages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def in_region(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Geographical areas where the word is used."
    word: Try "chips".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/inRegion"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def region_of(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words used in the specified geographical area."
    word: Try "canada".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/regionOf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pertains_to(words: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Words to which the original word is relevant."
    words: Try ".22-caliber".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{words}/pertainsTo"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pronunciation(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "How to pronounce a word, according to the International Phonetic Alphabet. May include multiple results if the word is pronounced differently depending on its part of speech."
    word: Try "wind". It sounds different depending on if its a noun or a verb.
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/pronunciation"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def syllables(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the word broken down into syllables."
    word: Try "incredible".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/syllables"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def frequency(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Expands upon the frequeny score returned by the main /words/{word} endpoint.  Returns zipf, a score indicating how common the word is in the English language, with a range of 1 to 7; perMillion, the number of times the word is likely to appear in a corpus of one million English words; and diversity, a 0-1 scale the shows the likelyhood of the word appearing in an English document that is part of a corpus."
    word: Try "apartment"
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/frequency"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a random word, optionally matching a search criteria. You can use the same search criteria as the "Search" endpoint."
    
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/?random=true"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(pronunciationpattern: str='.*æm$', soundsmax: str=None, sounds: str=None, soundsmin: str=None, syllablesmax: str=None, limit: str='100', frequencymin: str=None, letterpattern: str='^a.{4}$', page: str='1', partofspeech: str=None, lettersmin: str=None, letters: str=None, lettersmax: str=None, syllablesmin: str=None, syllables: str=None, frequencymax: str=None, hasdetails: str='hasDetails', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for words matching the parameters you provide.  For more examples, please see the documentation on the website.  https://www.wordsapi.com/docs#search"
    pronunciationpattern: Find words whose pronunciation matches the regular expression.
        soundsmax: The maximum number of phonemes (sounds) the word can have.
        sounds: The number of phonemes (sounds) the word mush have.
        soundsmin: The minimum number of phonemes (sounds) the word can have.
        syllablesmax: The maximum number of syllables the word can have.
        limit: The number of results to return per page. Must be between 1 and 100. Default is 100.
        frequencymin: The minimum frequency score of words to return. You can use this to limit your search to words that people are familiar with (like "go", with a frequency of 6.98). The range is from 1.74 - 8.03.
        letterpattern: Find words whose letters match the regular expression.
        page: The page of results to return. The default is page 1.
        partofspeech: The matching word must have at least one definition with this part of speech.
        lettersmin: The minimum number of letters the word must have.
        letters: The number of letters the word must have.
        lettersmax: The maximum number of letters the word can have.
        syllablesmin: The minimum number of syllables the word can have.
        syllables: The number of syllables the word must have.
        frequencymax: The maximum frequency score of words to return. You can use this to limit your search to words that aren't seen as frequently (like "zygote", with a frequency of 2.55). The range is from 1.74 - 8.03.
        hasdetails: A comma delimited list of detail types the word has. For instance, to find words that have both "typeOf" and "hasCategories" relationship, you would send "hasDetails=typeOf,hasCategories".
        
    """
    url = f"https://wordsapiv1.p.rapidapi.com/words/"
    querystring = {}
    if pronunciationpattern:
        querystring['pronunciationpattern'] = pronunciationpattern
    if soundsmax:
        querystring['soundsmax'] = soundsmax
    if sounds:
        querystring['sounds'] = sounds
    if soundsmin:
        querystring['soundsMin'] = soundsmin
    if syllablesmax:
        querystring['syllablesMax'] = syllablesmax
    if limit:
        querystring['limit'] = limit
    if frequencymin:
        querystring['frequencymin'] = frequencymin
    if letterpattern:
        querystring['letterPattern'] = letterpattern
    if page:
        querystring['page'] = page
    if partofspeech:
        querystring['partofspeech'] = partofspeech
    if lettersmin:
        querystring['lettersmin'] = lettersmin
    if letters:
        querystring['letters'] = letters
    if lettersmax:
        querystring['lettersMax'] = lettersmax
    if syllablesmin:
        querystring['syllablesMin'] = syllablesmin
    if syllables:
        querystring['syllables'] = syllables
    if frequencymax:
        querystring['frequencymax'] = frequencymax
    if hasdetails:
        querystring['hasDetails'] = hasdetails
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

