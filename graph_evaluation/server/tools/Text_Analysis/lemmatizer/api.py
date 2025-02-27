import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def extract_get(text: str, flag: str=None, exclude_non_content_words: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the root of a word or roots of a string of words."
    text: Enter some text to extract roots (maximum 200 words or 3,000 characters)
        flag: Pass an optional flag to change output. If "VALID_TOKENS_ONLY" is passed, an array list is returned with only valid lemmas in the order found in the string passed in the text parameter. If "VALID_TOKENS_ONLY_ORDER_BY_OCCURRENCE" is passed, an array list is returned with only valid lemmas in the order of occurrence with the most occurring lemma first. If "VALID_TOKENS_ONLY_ORDER_BY_OCCURRENCE_SHOW_COUNT" (default) is passed, an array list is returned with only valid lemmas in the order of occurrence with the most occurring lemma first, but with the lemma in the key and the count in the value. If "ALL_TOKENS" is passed, an array list is returned with all the words in the string passed in the text parameter with words that could be lemmatized as lemmas and words that could not be lemmatized left as is. If "ALL_TOKENS_INVALID_LOWERCASED" is passed, an array list is returned with all the words in the string passed in the text parameter with words that could be lemmatized as lemmas and words that could not be lemmatized left as is, but lowercased. If "ALL_TOKENS_INVALID_EMPTY_STRING" is passed, an array list is returned with all the words in the string passed in the text parameter with words that could be lemmatized as lemmas and words that could not be lemmatized as empty-string items in the array. If "ALL_TOKENS_INVALID_NEGATIVE_ONE" is passed, an array list is returned with all the words in the string passed in the text parameter with words that could be lemmatized as lemmas and words that could not be lemmatized as -1. If "ALL_TOKENS_ORDER_BY_OCCURRENCE" is passed, an array list is returned with all the words in the string passed in the text parameter, with words that could be lemmatized as lemmas and words that could not be lemmatized left as is, all in the order of occurrence with the most occurring first. If "ALL_TOKENS_ORDER_BY_OCCURRENCE_SHOW_COUNT" is passed, an array list is returned with all the words in the string passed in the text parameter, with words that could be lemmatized as lemmas and words that could not be lemmatized left as is, all in the order of occurrence with the most occurring first, but with the words and lemmas in the key and the count in the value.
        exclude_non_content_words: Optional boolean to exclude non content words (ie. "the", "is", "take")
        
    """
    url = f"https://twinword-lemmatizer1.p.rapidapi.com/extract/"
    querystring = {'text': text, }
    if flag:
        querystring['flag'] = flag
    if exclude_non_content_words:
        querystring['exclude_non_content_words'] = exclude_non_content_words
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-lemmatizer1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

