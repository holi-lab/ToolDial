import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_articles(subject: str, top_n_keywords: int=10, max_articles: int=10, last_n_hours: int=48, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides the latest crypto news articles scraped from the web  incl. sentiment analysis ([textblob](https://textblob.readthedocs.io/en/dev/quickstart.html)) and keyword extraction ([Yake](http://yake.inesctec.pt/)). 
		
		For historical cryptonews data check out our open source dataset on [Kaggle](https://www.kaggle.com/oliviervha/crypto-news)
		
		Parameters:
		- **[REQUIRED] subject** e.g. bitcoin, altcoin, ethereum, nft, blockchain, defi
		- **[OPTIONAL] last_n_hours**: Only news articles from the last *n* hours are provided (default: 24).
		- **[OPTIONAL] max_articles**: Maximum number of articles in response (default:100, max:100)
		- **[OPTIONAL] top_n_keywords**: Number of keywords to extract from article (default:10)
		
		Currently news is scraped from:
		- [CryptoNews](cryptonews.com)
		- [CoinTelegraph](https://cointelegraph.com/)
		- [CryptoPotato](https://cryptopotato.com/)
		
		If you have suggestions for more sources to add, please get in contact.
		
		More information:
		- Get the [historical Crypto News dataset](https://www.kaggle.com/oliviervha/crypto-news) from Kaggle!
		- Click [here](https://app.swaggerhub.com/apis-docs/CryptoAdvisor/CryptoNews/1.0.0#/developers/cryptonews) for the API Swagger documentation.
		- Keep up-to-date with our [CryptoNews+ Twitter](https://twitter.com/cryptonews_plus) account!"
    top_n_keywords: Number of keywords to be extracted from the article
        max_articles: The maximum number of articles in response (default:100,max:100).
        last_n_hours: Crypto articles from the last *n* hours (default: 24 hrs)
        
    """
    url = f"https://crypto-news11.p.rapidapi.com/cryptonews/{subject}"
    querystring = {}
    if top_n_keywords:
        querystring['top_n_keywords'] = top_n_keywords
    if max_articles:
        querystring['max_articles'] = max_articles
    if last_n_hours:
        querystring['last_n_hours'] = last_n_hours
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_tweets(top_n_keywords: int=10, max_tweets: int=10, last_n_hours: int=4, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides the latest crypto news tweets incl. sentiment analysis ([textblob](https://textblob.readthedocs.io/en/dev/quickstart.html)) and keyword extraction ([Yake](http://yake.inesctec.pt/)). 
		
		Parameters:
		- **[OPTIONAL] last_n_hours**: Only tweets from the last *n* hours are provided (default: 12).
		- **[OPTIONAL] max_articles**: Maximum number of tweets in response (default:100, max:100)
		- **[OPTIONAL] top_n_keywords**: Number of keywords to extract form tweet (default:10)
		
		Currently, we make use of the following sources:
		- [@BTCTN](https://twitter.com/BTCTN) 
		- [@CryptoBoomNews](https://twitter.com/CryptoBoomNews)
		- [@cryptocom](https://twitter.com/cryptocom)
		- [@Crypto_Potato](https://twitter.com/crypto_potato)
		
		If you have any suggestions for twitter accounts to add, please get in touch."
    top_n_keywords: The number of keywords extracted from the article.
        max_tweets: Maximum number of tweets in response (default:100, max:100)
        last_n_hours: Parameter to get crypto tweets for the last *n* hours (default: 12)
        
    """
    url = f"https://crypto-news11.p.rapidapi.com/cryptotweets"
    querystring = {}
    if top_n_keywords:
        querystring['top_n_keywords'] = top_n_keywords
    if max_tweets:
        querystring['max_tweets'] = max_tweets
    if last_n_hours:
        querystring['last_n_hours'] = last_n_hours
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-news11.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

