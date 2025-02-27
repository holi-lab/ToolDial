import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def analyze_get(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return sentiment analysis results with score for the given text."
    text: Enter some text to analyze (maximum 100 words or 1,500 characters)
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/sentiment_analyze/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def association(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the associations of a word."
    entry: Type a word to get its associations
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/word_association/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classify_get(text: str, title: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Classify text into product categories or contact us to customize and use your own category sets. Enter some text to find its related product categories:"
    text: Enter some text to find related categories.
        title: Enter title of text (optional).
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/text_classify/"
    querystring = {'text': text, }
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def definition(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the definitions of a word."
    entry: Type a word to get its definitions
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/word_definition/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def definition_kr(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Korean definitions of a word."
    entry: Type a word to get its Korean definitions
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/word_definition_kr/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def difficulty(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the difficulty level of a word."
    entry: Type a word to get its difficulty level
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/word_difficulty/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exam_history(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "See which exams a word has been on"
    entry: Type a word to see which exams it has been on
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/word_examhistory/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def example(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "See examples of a word used in a sentence"
    entry: Type a word to see examples of it used in a sentence
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/word_example/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract_get(text: str, flag: str=None, exclude_non_content_words: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the root of a word or roots of a string of words."
    text: Enter some text to extract roots (maximum 200 words or 3,000 characters)
        flag: Pass an optional flag to change output. If "VALID_TOKENS_ONLY" is passed, an array list is returned with only valid lemmas in the order found in the string passed in the text parameter. If "VALID_TOKENS_ONLY_ORDER_BY_OCCURRENCE" is passed, an array list is returned with only valid lemmas in the order of occurrence with the most occurring lemma first. If "VALID_TOKENS_ONLY_ORDER_BY_OCCURRENCE_SHOW_COUNT" (default) is passed, an array list is returned with only valid lemmas in the order of occurrence with the most occurring lemma first, but with the lemma in the key and the count in the value. If "ALL_TOKENS" is passed, an array list is returned with all the words in the string passed in the text parameter with words that could be lemmatized as lemmas and words that could not be lemmatized left as is. If "ALL_TOKENS_INVALID_LOWERCASED" is passed, an array list is returned with all the words in the string passed in the text parameter with words that could be lemmatized as lemmas and words that could not be lemmatized left as is, but lowercased. If "ALL_TOKENS_INVALID_EMPTY_STRING" is passed, an array list is returned with all the words in the string passed in the text parameter with words that could be lemmatized as lemmas and words that could not be lemmatized as empty-string items in the array. If "ALL_TOKENS_INVALID_NEGATIVE_ONE" is passed, an array list is returned with all the words in the string passed in the text parameter with words that could be lemmatized as lemmas and words that could not be lemmatized as -1. If "ALL_TOKENS_ORDER_BY_OCCURRENCE" is passed, an array list is returned with all the words in the string passed in the text parameter, with words that could be lemmatized as lemmas and words that could not be lemmatized left as is, all in the order of occurrence with the most occurring first. If "ALL_TOKENS_ORDER_BY_OCCURRENCE_SHOW_COUNT" is passed, an array list is returned with all the words in the string passed in the text parameter, with words that could be lemmatized as lemmas and words that could not be lemmatized left as is, all in the order of occurrence with the most occurring first, but with the words and lemmas in the key and the count in the value.
        exclude_non_content_words: Optional boolean to exclude non content words (ie. "the", "is", "take")
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/lemma_extract/"
    querystring = {'text': text, }
    if flag:
        querystring['flag'] = flag
    if exclude_non_content_words:
        querystring['exclude_non_content_words'] = exclude_non_content_words
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_get(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect and generate human like topics to the given text."
    text: Enter some text to generate topics (maximum 200 words or 3,000 characters)
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/topic_generate/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recommend_get(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Recommend highly related categories for e-commerce and other uses."
    text: Enter some text to find related categories:
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/category_recommend/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reference(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the broad terms, narrow terms, related terms, evocations, synonyms, associations, and derived terms of a word."
    entry: Type a word to get its broad terms, narrow terms, related terms, evocations, synonyms, associations, and derived terms
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/word_reference/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def text_get(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Evaluate the difficulty level of a word, sentence, or paragraph."
    text: Input a some text to evaluate its difficulty level (maximum 200 words or 3,000 characters)
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/score_text/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def text_similarity_get(text1: str, text2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Evaluate the similarity of two words, sentences, or paragraphs."
    text1: Input the first text to compare.
        text2: Input a second text to compare its similarity with the first text.
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/text_similarity/"
    querystring = {'text1': text1, 'text2': text2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def theme(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the themes of a word."
    entry: Type a word to get its themes
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/word_theme/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_1(level: int, area: str='sat', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Customized word association quiz for game and e-learning software."
    level: Select the difficulty level of the test (1 - 10)
        area: Select a test to generate quiz questions and answers (es, ms, hs, ksat, toeic, toefl, teps, sat, ielts, gre, gmat, overall)
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/quiz_type1/"
    querystring = {'level': level, }
    if area:
        querystring['area'] = area
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def word_get(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Evaluate the difficulty level of a word."
    entry: Input a word to evaluate its difficulty level
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/score_word/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def word_associations_get(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get word associations with semantic distance score."
    entry: A word, phrase, or paragraph.
        
    """
    url = f"https://twinword-twinword-bundle-v1.p.rapidapi.com/word_associations/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-twinword-bundle-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

