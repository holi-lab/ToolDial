import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news(country: str='US', language: str='en', period: str='1d', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "news" endpoint is a RESTful API endpoint for retrieving the latest news articles. The endpoint accepts the following parameters:
		
		language: specifies the language of the news articles to be returned (default: "en").
		country: specifies the country for which news articles are to be retrieved (default: "US").
		period: specifies the time frame of the news articles to be retrieved (default: "1d"). The format of the timeframe is a string comprised of a number, followed by a letter representing the time operator. For example 1y would signify 1 year. Full list of operators are: h = hours (eg: 12h), d = days (eg: 7d), m = months (eg: 6m), y = years (eg: 1y).
		max_results: specifies the maximum number of news articles to be returned (default: 10).
		The API returns the latest news articles as a JSON object, containing an array of news items, each with the following fields:
		**
		headline**: title of the news article.
		**link:** URL of the news article.
		**publish_date: **publication date of the news article.
		**publisher**: name of the publisher of the news article.
		**news_body:** a short description of the news article.
		
		The API call can be made by sending a GET request to the API endpoint, along with the desired parameters. The API returns a JSON object containing an array of the latest news articles."
    country: The country parameter is an optional query parameter for the API endpoint /news that specifies the country for which the news should be fetched. The default value for this parameter is US if no value is provided. The value of this parameter should be a two letter country code as per the ISO 3166-1 Alpha-2 standard.

For example, if you want to fetch news for India, the API endpoint can be /news?country=IN.

The API fetches news articles from Google News and provides a JSON response that contains the list of news articles along with the details of each article. The country parameter helps to filter the news articles based on the specified country.

Available Countries are:
'Australia': 'AU', 'Botswana': 'BW', 'Canada ': 'CA', 'Ethiopia': 'ET', 'Ghana': 'GH', 'India ': 'IN',
 'Indonesia': 'ID', 'Ireland': 'IE', 'Israel ': 'IL', 'Kenya': 'KE', 'Latvia': 'LV', 'Malaysia': 'MY', 'Namibia': 'NA',
 'New Zealand': 'NZ', 'Nigeria': 'NG', 'Pakistan': 'PK', 'Philippines': 'PH', 'Singapore': 'SG', 'South Africa': 'ZA',
 'Tanzania': 'TZ', 'Uganda': 'UG', 'United Kingdom': 'GB', 'United States': 'US', 'Zimbabwe': 'ZW',
 'Czech Republic': 'CZ', 'Germany': 'DE', 'Austria': 'AT', 'Switzerland': 'CH', 'Argentina': 'AR', 'Chile': 'CL',
 'Colombia': 'CO', 'Cuba': 'CU', 'Mexico': 'MX', 'Peru': 'PE', 'Venezuela': 'VE', 'Belgium ': 'BE', 'France': 'FR',
 'Morocco': 'MA', 'Senegal': 'SN', 'Italy': 'IT', 'Lithuania': 'LT', 'Hungary': 'HU', 'Netherlands': 'NL',
 'Norway': 'NO', 'Poland': 'PL', 'Brazil': 'BR', 'Portugal': 'PT', 'Romania': 'RO', 'Slovakia': 'SK', 'Slovenia': 'SI',
 'Sweden': 'SE', 'Vietnam': 'VN', 'Turkey': 'TR', 'Greece': 'GR', 'Bulgaria': 'BG', 'Russia': 'RU', 'Ukraine ': 'UA',
 'Serbia': 'RS', 'United Arab Emirates': 'AE', 'Saudi Arabia': 'SA', 'Lebanon': 'LB', 'Egypt': 'EG',
 'Bangladesh': 'BD', 'Thailand': 'TH', 'China': 'CN', 'Taiwan': 'TW', 'Hong Kong': 'HK', 'Japan': 'JP',
 'Republic of Korea': 'KR'
        language: The 'language' parameter of the API is a query parameter that is used to specify the language in which the news articles are to be returned. The parameter is optional and its default value is \"en\".

Available Languages are:
'english': 'en', 'indonesian': 'id', 'czech': 'cs', 'german': 'de', 'spanish': 'es-419', 'french': 'fr',
 'italian': 'it', 'latvian': 'lv', 'lithuanian': 'lt', 'hungarian': 'hu', 'dutch': 'nl', 'norwegian': 'no',
 'polish': 'pl', 'portuguese brasil': 'pt-419', 'portuguese portugal': 'pt-150', 'romanian': 'ro', 'slovak': 'sk',
 'slovenian': 'sl', 'swedish': 'sv', 'vietnamese': 'vi', 'turkish': 'tr', 'greek': 'el', 'bulgarian': 'bg',
 'russian': 'ru', 'serbian': 'sr', 'ukrainian': 'uk', 'hebrew': 'he', 'arabic': 'ar', 'marathi': 'mr', 'hindi': 'hi',
 'bengali': 'bn', 'tamil': 'ta', 'telugu': 'te', 'malyalam': 'ml', 'thai': 'th', 'chinese simplified': 'zh-Hans',
 'chinese traditional': 'zh-Hant', 'japanese': 'ja', 'korean': 'ko'
        period: The period parameter is used to specify the timeframe for which the news articles will be fetched. The format of the period string is comprised of a number, followed by a letter representing the time operator. For example, 1y would signify 1 year. The available time operators are as follows:

h represents hours (eg: 12h)
d represents days (eg: 7d)
m represents months (eg: 6m)
y represents years (eg: 1y)
        max_results: The \"max_results\" parameter specifies the maximum number of news articles to be returned in the response. This parameter accepts an integer value and the default value is set to 10. The number entered in this parameter must be less than or equal to 100, as the API has a limit of returning maximum 100 news articles in a single request. The \"max_results\" parameter allows you to control the number of news articles returned in a single request, which can be useful for optimizing your application's performance.
        
    """
    url = f"https://best-news-fetching-api.p.rapidapi.com/"
    querystring = {}
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    if period:
        querystring['period'] = period
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-news-fetching-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

