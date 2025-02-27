import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def summarize(text: str, max_words: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Summarize a large document without missing key details"
    
    """
    url = f"https://entity-and-sentiment-extractor.p.rapidapi.com/summarize"
    querystring = {'text': text, }
    if max_words:
        querystring['max_words'] = max_words
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entity-and-sentiment-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def entity_non_english_entity_recognition(text: str='Apple 对每个人来说都是一家伟大的公司。你怎么认为？三星也有同样的想法。', lang_code: str='zh', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently supports the following major languages:
		Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish which correspond to the following lang_code
		['zh', 'zh-Hant', 'en', 'fr', 'de', 'it', 'ja', 'ko', 'pt', 'ru', 'es']"
    
    """
    url = f"https://entity-and-sentiment-extractor.p.rapidapi.com/entities"
    querystring = {}
    if text:
        querystring['text'] = text
    if lang_code:
        querystring['lang_code'] = lang_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entity-and-sentiment-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def entity_list_for_each_sentiment_types(text: str, content_type: str, only_entities_and_sentiments: str='true', positive_threshold: str='0.5', lang_code: str='en', negative_threshold: str='-0.3', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Identify various entities and their sentiment in a document.
		
		It will also return a wiki link for popular entities.
		
		Notice the **salience** score in the example output.
		The salience score tells the which entity this document is about. The higher the score of an entity, the more indication that the document is talking about that particular entity.
		
		
		API can accept HTML input as well.
		Please try following:
		```
		content_type=HTML
		text=<p>I like a human like entity extractor, which Apple, iPhone had too</p>
		```
		
		**lang_code** is an optional param. If set it will speed up the response.
		Use https://en.wikipedia.org/wiki/ISO_639-1 codes for languages
		
		**stock_lookup** is an optional param. Set it "true" if you also need the current stock info for any extracted company (remember that the stock ticker gets added/delisted from exchanges). Enabling stock_lookup will slow down the response a bit."
    content_type: plain_text or html
        lang_code: set lang_code='NA' if don't know the language of the text. Otherwise, it will default to en (English)
        
    """
    url = f"https://entity-and-sentiment-extractor.p.rapidapi.com/entities_with_sentiment"
    querystring = {'text': text, 'content_type': content_type, }
    if only_entities_and_sentiments:
        querystring['only_entities_and_sentiments'] = only_entities_and_sentiments
    if positive_threshold:
        querystring['positive_threshold'] = positive_threshold
    if lang_code:
        querystring['lang_code'] = lang_code
    if negative_threshold:
        querystring['negative_threshold'] = negative_threshold
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entity-and-sentiment-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sentence_top_positive_negative(text: str, content_type: str='plain_text', lang_code: str='en', stock_lookup: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API can accept HTML input as well.
		Please try following:
		```
		content_type=HTML
		text=<p>I like a human like entity extractor, which Apple, iPhone had too</p>
		```
		
		**lang_code** is an optional param. If set it will speed up the response.
		Use https://en.wikipedia.org/wiki/ISO_639-1 codes for languages
		
		**stock_lookup** is an optional param. Set it "true" if you also need the current stock info for any extracted company (remember that the stock ticker gets added/delisted from exchanges). Enabling stock_lookup will slow down the response a bit."
    content_type: plain_text or html
        lang_code: set lang_code='NA' if don't know the language of the text. Otherwise, it will default to en (English)
        
    """
    url = f"https://entity-and-sentiment-extractor.p.rapidapi.com/top_sentiment_sentences"
    querystring = {'text': text, }
    if content_type:
        querystring['content_type'] = content_type
    if lang_code:
        querystring['lang_code'] = lang_code
    if stock_lookup:
        querystring['stock_lookup'] = stock_lookup
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entity-and-sentiment-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sentence_sentiment_analysis(text: str, lang_code: str='en', stock_lookup: str='True', content_type: str='plain_text', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API can accept HTML input as well.
		Please try following:
		```
		content_type=HTML
		text=<p>I like a human like entity extractor, which Apple, iPhone had too</p>
		```
		
		**lang_code** is an optional param. If set it will speed up the response.
		Use https://en.wikipedia.org/wiki/ISO_639-1 codes for languages
		
		**stock_lookup** is an optional param. Set it "true" if you also need the current stock info for any extracted company (remember that the stock ticker gets added/delisted from exchanges). Enabling stock_lookup will slow down the response a bit."
    lang_code: set lang_code='NA' if don't know the language of the text. Otherwise, it will default to en (English)
        content_type: plain_text or html
        
    """
    url = f"https://entity-and-sentiment-extractor.p.rapidapi.com/sentiment"
    querystring = {'text': text, }
    if lang_code:
        querystring['lang_code'] = lang_code
    if stock_lookup:
        querystring['stock_lookup'] = stock_lookup
    if content_type:
        querystring['content_type'] = content_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entity-and-sentiment-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def entity_detailed_analysis(text: str, content_type: str='plain_text', lang_code: str='en', stock_lookup: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Identify various entities and their sentiment in a document.
		
		It will also return a wiki link for popular entities.
		
		Notice the **salience** score in the example output.
		The salience score tells the which entity this document is about. The higher the score of an entity, the more indication that the document is talking about that particular entity.
		
		
		API can accept HTML input as well.
		Please try following:
		```
		content_type=HTML
		text=<p>I like a human like entity extractor, which Apple, iPhone had too</p>
		```
		
		**lang_code** is an optional param. If set it will speed up the response.
		Use https://en.wikipedia.org/wiki/ISO_639-1 codes for languages
		
		**stock_lookup** is an optional param. Set it "true" if you also need the current stock info for any extracted company (remember that the stock ticker gets added/delisted from exchanges). Enabling stock_lookup will slow down the response a bit."
    content_type: plain_text or html
        lang_code: set lang_code='NA' if don't know the language of the text. Otherwise, it will default to en (English)
        
    """
    url = f"https://entity-and-sentiment-extractor.p.rapidapi.com/entities"
    querystring = {'text': text, }
    if content_type:
        querystring['content_type'] = content_type
    if lang_code:
        querystring['lang_code'] = lang_code
    if stock_lookup:
        querystring['stock_lookup'] = stock_lookup
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entity-and-sentiment-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

