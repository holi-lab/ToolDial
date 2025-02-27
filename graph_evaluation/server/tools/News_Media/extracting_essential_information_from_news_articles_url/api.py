import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def to_extract_the_authors_who_authored_the_news_article(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint is more applicable to be pointed to news articles on the web. However, this API will be able to extract appropriate information from a regular web page as well.
		/getAuthors => To extract the authors who authored the news article
		e.g.) https://web.pregnya.com/getAuthors?url=https://www.nytimes.com/2020/03/21/arts/d-nice-instagram.html"
    
    """
    url = f"https://extracting-essential-information-from-news-articles-url.p.rapidapi.com/getAuthors"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "extracting-essential-information-from-news-articles-url.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def to_guess_and_extract_the_date_of_news_article_or_web_page(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint is more applicable to be pointed to news articles on the web. However, this API will be able to extract appropriate information from a regular web page as well.
		guessDateforArticle  => Guess the date of News article or web page
		e.g.) https://web.pregnya.com/guessDateforArticle?url=https://www.nytimes.com/2020/03/21/arts/d-nice-instagram.html"
    
    """
    url = f"https://extracting-essential-information-from-news-articles-url.p.rapidapi.com/guessDateforArticle"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "extracting-essential-information-from-news-articles-url.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract_essential_and_comprehensive_information_from_news_article(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "These API endpoints are more applicable to be pointed to news articles on the web.
		
		1) /extractfromArticle => To extract most of the essential and comprehensive elements meta tags/data, date, author,description, summary, natural language processing info, Entities (like PEOPLE, PLACES, THINGS, EVENT, ORGANIZATION, etc) and related images.
		e.g) https://web.pregnya.com/extractfromArticle?url=https://www.latimes.com/california/story/2020-05-20/defying-state-order-thousands-of-pastors-say-they-plan-to-hold-in-person-services-for-the-pentecost"
    
    """
    url = f"https://extracting-essential-information-from-news-articles-url.p.rapidapi.com/extractfromArticle"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "extracting-essential-information-from-news-articles-url.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def to_extract_the_text_or_description_of_the_news_article(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint is more applicable to be pointed to news articles on the web. However, this API will be able to extract appropriate information from a regular web page as well.
		/getText  => To extract the Text or description of the news article
		e.g.) https://web.pregnya.com/getText?url=https://www.nytimes.com/2020/03/21/arts/d-nice-instagram.html"
    
    """
    url = f"https://extracting-essential-information-from-news-articles-url.p.rapidapi.com/getText"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "extracting-essential-information-from-news-articles-url.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def to_extract_only_the_meta_tags_like_meta_image_meta_title_meta_keyword_met_description_meta_language_meta_favicon(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint is more applicable to be pointed to news articles on the web. However, this API will be able to extract appropriate information from a regular web page as well.
		/getMETATags => To extract only the meta tags like meta image, meta title, meta keyword, met description, meta language, & meta favicon
		e.g.) https://web.pregnya.com/getMETATags?url=https://www.nytimes.com/2020/03/21/arts/d-nice-instagram.html"
    
    """
    url = f"https://extracting-essential-information-from-news-articles-url.p.rapidapi.com/getMETATags"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "extracting-essential-information-from-news-articles-url.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def to_extract_the_title_of_the_news_article(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint is more applicable to be pointed to news articles on the web. However, this API will be able to extract appropriate information from a regular web page as well.
		/getTitle  => To extract the TITLE of the news article
		e.g.) https://web.pregnya.com/getTitle?url=https://www.nytimes.com/2020/03/21/arts/d-nice-instagram.html"
    
    """
    url = f"https://extracting-essential-information-from-news-articles-url.p.rapidapi.com/getTitle"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "extracting-essential-information-from-news-articles-url.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract_natural_language_processing_nlp_information_like_keywords_summary_and_entities(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint is more applicable to be pointed to news articles on the web. However, this API will be able to extract appropriate information from a regular web page as well.
		/extractNLP  => To extract Natural Language Processing (NLP) information like keywords, summary, and entities
		e.g.) https://web.pregnya.com/extractNLP?url=https://www.nbcnews.com/news/us-news/chicago-experiences-most-violent-memorial-day-weekend-five-years-n1214991"
    
    """
    url = f"https://extracting-essential-information-from-news-articles-url.p.rapidapi.com/extractNLP"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "extracting-essential-information-from-news-articles-url.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

