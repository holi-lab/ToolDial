import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_predictive_spikes_by_coin_name(coinname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve predictive social media spikes in the last 24 hours by Coin Name. Results will contain Sentiment, Weighted Sentiment, Mentions and Followers spikes both positive and negative. 
		
		This brings predictive spikes (that went through another machine learning model to predict likelihood of market change within the next 24 hours). These spikes are colored as red and orange.
		
		**Input:**
		
		- Coin Name (Required)
		
		**Output:**
		- EventId
		- EventType
		- Event Time
		- Related coins
		- Event Color
		- Event text
		
		**Definitions:**
		
		- Related coins: Coins mentioned in the post
		- Event type: Determines the type of spike (Sentiment, Weighted sentiment, Mentions or Followers) and the timeframe of the spike (1 hour, 8 hours or 1 day)
		- Event Color: Shows the magnitude of the event (From most important to less important: Red, Orange, Yellow, Gray and Black)
		- Mentions: Number of posts related to the coin in tracked social media sources.
		- Sentiment: Democratic Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative). [Democratic Sentiment is a sentiment that gives the same weight to each mention related to the coin despite the number of followers for each mention]
		- WeightedSentiment: Weighted Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative). [ Weighted Sentiment is a sentiment that gives more weight to mentions with more followers]
		- Followers: Sum of followers reached by coin related mentions
		
		**Details:**
		
		- Results are up to 100
		- All dates are UTC
		- Coins in this endpoint are 3 Billion market cap and more"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/socialnews/getpredictivespikesbycoinname"
    querystring = {'coinName': coinname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recent_news_by_coin_ticker(cointicker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Recent News By Coin Ticker"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/socialnews/getrecentnewsbycointicker"
    querystring = {'coinTicker': cointicker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recent_news_by_coin_name(coinname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Recent News By Coin Name"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/socialnews/getrecentnewsbycoinname"
    querystring = {'coinName': coinname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recent_updates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Recent Updates"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/socialnews/getrecentnews"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_social_spikes_by_coin_ticker(cointicker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve social media spikes in the last 24 hours by Coin Ticker. Results will contain Sentiment, Weighted Sentiment, Mentions and Followers spikes both positive and negative. 
		
		This brings all spikes including:
		
		- Predictive spikes (that went through another machine learning model to predict likelihood of market change within the next 24 hours). These spikes are colored as red and orange.
		- Other spikes in social media metrics that did not go through another layer of market moving prediction. These include spikes that are colored as yellow, gray, and black.
		
		**Input:**
		
		- Coin Ticker (Required)
		
		**Output:**
		
		- EventId
		- EventType
		- Event Time
		- Related coins
		- Event Color
		- Event text
		
		**Definitions:**
		
		- Related coins: Coins mentioned in the post
		- Event type: Determines the type of spike (Sentiment, Weighted sentiment, Mentions or Followers) and the timeframe of the spike (1 hour, 8 hours or 1 day)
		- Event Color: Shows the magnitude of the event (From most important to less important: Red, Orange, Yellow, Gray and Black)
		- Mentions: Number of posts related to the coin in tracked social media sources.
		- Sentiment: Democratic Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative). [Democratic Sentiment is a sentiment that gives the same weight to each mention related to the coin despite the number of followers for each mention]
		- WeightedSentiment: Weighted Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative). [ Weighted Sentiment is a sentiment that gives more weight to mentions with more followers]
		- Followers: Sum of followers reached by coin related mentions
		
		**Details:**
		
		- Results are up to 100
		- All dates are UTC"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/socialnews/getsocialspikesbycointicker"
    querystring = {'coinTicker': cointicker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_predictive_spikes_by_coin_ticker(cointicker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve predictive social media spikes in the last 24 hours by Coin Ticker. Results will contain Sentiment, Weighted Sentiment, Mentions and Followers spikes both positive and negative. 
		
		This brings predictive spikes (that went through another machine learning model to predict likelihood of market change within the next 24 hours). These spikes are colored as red and orange.
		
		**Input:**
		
		- Coin Ticker (Required)
		
		**Output:**
		
		- EventId
		- EventType
		- Event Time
		- Related coins
		- Event Color
		- Event text
		
		
		**Definitions:**
		
		- Related coins: Coins mentioned in the post
		- Event type: Determines the type of spike (Sentiment, Weighted sentiment, Mentions or Followers) and the timeframe of the spike (1 hour, 8 hours or 1 day)
		- Event Color: Shows the magnitude of the event (From most important to less important: Red, Orange, Yellow, Gray and Black)
		- Mentions: Number of posts related to the coin in tracked social media sources.
		- Sentiment: Democratic Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative). [Democratic Sentiment is a sentiment that gives the same weight to each mention related to the coin despite the number of followers for each mention]
		- WeightedSentiment: Weighted Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative). [ Weighted Sentiment is a sentiment that gives more weight to mentions with more followers]
		- Followers: Sum of followers reached by coin related mentions
		
		**Details:**
		
		- Results are up to 100
		- All dates are UTC
		- Coins in this endpoint are 3 Billion market cap and more"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/socialnews/getpredictivespikesbycointicker"
    querystring = {'coinTicker': cointicker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_social_spikes_by_coin_name(coinname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve social media spikes in the last 24 hours by Coin Name. Results will contain Sentiment, Weighted Sentiment, Mentions and Followers spikes both positive and negative. 
		
		This brings all spikes including:
		
		- Predictive spikes (that went through another machine learning model to predict likelihood of market change within the next 24 hours). These spikes are colored as red and orange.
		- Other spikes in social media metrics that did not go through another layer of market moving prediction. These include spikes that are colored as yellow, gray, and black.
		
		**Input:**
		
		- Coin Name (Required)
		
		**Output:**
		
		- EventId
		- EventType
		- Event Time
		- Related coins
		- Event Color
		- Event text
		
		**Definitions:**
		
		- Related coins: Coins mentioned in the post
		- Event type: Determines the type of spike (Sentiment, Weighted sentiment, Mentions or Followers) and the timeframe of the spike (1 hour, 8 hours or 1 day)
		- Event Color: Shows the magnitude of the event (From most important to less important: Red, Orange, Yellow, Gray and Black)
		- Mentions: Number of posts related to the coin in tracked social media sources.
		- Sentiment: Democratic Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative). [Democratic Sentiment is a sentiment that gives the same weight to each mention related to the coin despite the number of followers for each mention]
		- WeightedSentiment: Weighted Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative). [ Weighted Sentiment is a sentiment that gives more weight to mentions with more followers]
		- Followers: Sum of followers reached by coin related mentions
		
		**Details:**
		
		- Results are up to 100
		- All dates are UTC"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/socialnews/getsocialspikesbycoinname"
    querystring = {'coinName': coinname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_daily_influencer_posts_by_coin_name(coinname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve news and posts from Top Influencers in social media ordered by date by coin name
		This endpoint contains results on a specific coin. To search for influencer posts on all coins coins check "Get Daily Influencer Posts"
		
		**Input:**
		
		- Coin Name (Required)
		
		
		**Output:**
		
		- EventId
		- EventType
		- Event Time
		- Publisher
		- Source
		- Related coins
		- Event Color
		- Number of followers
		- Event text
		
		**Definitions:**
		
		- Publisher: User name of the publisher of the event in social media
		- Source: Social media where the post was published
		- Related coins: Coins mentioned in the post
		- Event Color: Shows the magnitude of the event (From most important to less important: Red, Orange, Yellow, Gray and Black)
		- Number of followers: Influencer´s number of followers in social media
		
		**Details:**
		
		- Results are up to 100
		- All dates are UTC"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/socialnews/getinfluencersnewsbycoinname"
    querystring = {'coinName': coinname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_daily_influencer_posts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve news and posts from Top Influencers in social media ordered by date
		This endpoint contains results from all coins. To search for posts on a specific coins check "Get Daily Influencer Posts by Coin Name" or "Get Daily Influencer Posts by Coin Ticker"
		
		**Output:**
		
		- EventId 
		- EventType
		- Event Time
		- Publisher
		- Source
		- Related coins
		- Event Color
		- Number of followers
		- Event text
		
		**Definitions:** 
		
		- Publisher: User name of the publisher of the event in social media
		- Source: Social media where the post was published
		- Related coins: Coins mentioned in the post
		- Event Color: Shows the magnitude of the event (From most important to less important: Red, Orange, Yellow, Gray and Black)
		- Number of followers:  Influencer´s number of followers in social media
		
		**Details:**
		
		- Results are up to 100
		- All dates are UTC"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/socialnews/getinfluencersnews"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_coins_by_followers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve real time top 10 coins by Followers
		
		**Output:**
		Each item contains:
		- Rank number
		- Coin name
		- Coin ticker
		- Followers value
		
		**Definitions:**
		- Followers: Sum of followers reached by coin related mentions
		
		**Details:**
		- All results are in real time"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/toptrendingcoins/gettopcoinsbyfollowers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_coins_by_sentiment(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve real time top 10 coins by Sentiment
		
		**Output:**
		Each item contains:
		- Rank number
		- Coin name
		- Coin ticker
		- Sentiment value
		
		**Definitions:**
		
		- Sentiment: Democratic Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative). Democratic Sentiment is a sentiment that gives the same weight to each mention related to the coin despite the number of followers for each mention]
		
		**Details:**
		- All results are in real time"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/toptrendingcoins/gettopcoinsbysentiment"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_daily_influencer_posts_by_coin_ticker(cointicker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve news and posts from Top Influencers in social media ordered by date by coin ticker
		This endpoint contains results on a specific coin. To search for influencer posts on all coins coins check "Get Daily Influencer Posts"
		
		**Input:**
		
		- Coin Ticker (Required)
		
		**Output:**
		
		- EventId
		- EventType
		- Event Time
		- Publisher
		- Source
		- Related coins
		- Event Color
		- Number of followers
		- Event text
		
		**Definitions:**
		
		- Publisher: User name of the publisher of the event in social media
		- Source: Social media where the post was published
		- Related coins: Coins mentioned in the post
		- Event Color: Shows the magnitude of the event (From most important to less important: Red, Orange, Yellow, Gray and Black)
		- Number of followers: Influencer´s number of followers in social media
		
		**Details:**
		
		- Results are up to 100
		- All dates are UTC"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/socialnews/getinfluencersnewsbycointicker"
    querystring = {'coinTicker': cointicker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_metrics_by_name(securityname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve  coin´s real time social media metrics by Coin name
		In the case you want to retrieve historical data check "Get Coin Historical Metrics by Name" or "Get Coin Historical Metrics by Ticker"
		
		**Input:** 
		- Security Name: Coin Name (Required)
		
		**Output:**
		
		- Sentiment
		- Sentiment change
		- Followers
		- Followers change
		- Mentions
		- Mentions change
		- Security ticker
		- Security name
		- DataTimeStamp
		
		**Definitions:**
		
		•	Mentions - Number of posts related to the coin in tracked social media sources in the requested date with TimeFrame required.
		•	MentionsChange - Percentage change of Mentions in the requested date vs the average of the same metric in the past 14 days.
		•	Sentiment - Democratic Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative)in the requested date with TimeFrame required.  [Democratic Sentiment is a sentiment that gives the same weight to each mention related to the coin despite the number of followers for each mention]
		•	SentimentChange - Percentage change of Democratic Sentiment in the requested date vs the average of the same metric in the past 14 days.
		•	Followers - Sum of followers reached by coin related mentions in the requested date with TimeFrame required.
		•	FollowersChange - Percentage change of Followers Reach in the requested date vs the average of the same metric in the past 14 days.
		
		**Details:**
		- All dates are in UTC
		- Results are on 1 day timeframe"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/tokenmetrics/getcoinmetricsbyname"
    querystring = {'securityName': securityname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_metrics_by_ticker(securityticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve coin´s  real time social media metrics by Coin ticker
		In the case you want to retrieve historical data check "Get Coin Historical Metrics by Name" or "Get Coin Historical Metrics by Ticker"
		
		**Input:** 
		- Security Ticker: Coin Ticker (Required)
		
		**Output:**
		
		- Sentiment
		- Sentiment change
		- Followers
		- Followers change
		- Mentions
		- Mentions change
		- Security ticker
		- Security name
		- DataTimeStamp
		
		**Definitions:**
		
		•	Mentions - Number of posts related to the coin in tracked social media sources in the requested date with TimeFrame required.
		•	MentionsChange - Percentage change of Mentions in the requested date vs the average of the same metric in the past 14 days.
		•	Sentiment - Democratic Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative)in the requested date with TimeFrame required.  [Democratic Sentiment is a sentiment that gives the same weight to each mention related to the coin despite the number of followers for each mention]
		•	SentimentChange - Percentage change of Democratic Sentiment in the requested date vs the average of the same metric in the past 14 days.
		•	Followers - Sum of followers reached by coin related mentions in the requested date with TimeFrame required.
		•	FollowersChange - Percentage change of Followers Reach in the requested date vs the average of the same metric in the past 14 days.
		
		**Details:**
		
		- All dates are in UTC
		- Results are on 1 day timeframe"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/tokenmetrics/getcoinmetricsbyticker"
    querystring = {'securityTicker': securityticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_coins_by_followers_change(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve real time top 10 coins by Mentions change in the last 24 hours
		
		**Output:**
		Each item contains:
		- Rank number
		- Coin name
		- Coin ticker
		- Followers change value
		
		**Definitions:**
		
		- FollowersChange: Percentage change of Followers Reach in the requested date vs the average of the same metric in the past 14 days
		
		
		**Details:**
		- All results are in real time"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/toptrendingcoins/gettopcoinsbyfollowerschange"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_coins_by_mentions_change(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve real time top 10 coins by Mentions change in the last 24 hours
		
		**Output:**
		Each item contains:
		- Rank number
		- Coin name
		- Coin ticker
		- Mentions change value
		
		**Definitions:**
		- MentionsChange: Percentage change of Mentions in the requested date vs the average of the same metric in the past 14 days.
		
		**Details:**
		- All results are in real time"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/toptrendingcoins/gettopcoinsbymentionschange"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_coins_by_mentions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve real time top 10 coins by Mentions
		
		**Output:**
		Each item contains:
		- Rank number
		- Coin name
		- Coin ticker
		- Mentions value
		
		**Definitions:**
		- Mentions: Number of posts related to the coin in tracked social media sources in the requested date with TimeFrame required.
		
		**Details:**
		- All results are in real time"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/toptrendingcoins/gettopcoinsbymentions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_coins_by_sentiment_change(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve real time top 10 coins by Sentiment change in the last 24 hours
		
		**Output:**
		Each item contains:
		- Rank number
		- Coin name
		- Coin ticker
		- Sentiment change value
		
		**Definitions:**
		- Sentiment Change: Percentage change of Democratic Sentiment in the requested date vs the average of the same metric in the past 14 days.
		
		**Details:**
		- All results are in real time"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/toptrendingcoins/gettopcoinsbysentimentchange"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_historical_metrics_by_name(securityname: str, date: str, timeframe: str='1D', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve  coin´s historical social media metrics by Coin Name
		In the case you want to retrieve real time data check "Get Coin Metrics by Name" or "Get Coin Metrics by Ticker"
		
		**Input:** 
		-  Date (MM/DD/YYYY HH:HH AM/PM) (Required)
		- Security Name: Coin Name (Required)
		- TimeFrame: 1d, 1h, 8h (Optional)
		
		**Output:**
		
		- Sentiment
		- Sentiment change
		- Weighted Sentiment
		- Weighted sentiment change
		- Followers
		- Followers change
		- Mentions
		- Mentions change
		- Security ticker
		- Security Name
		- TimeFrame
		- DataTimeStamp
		
		**Definitions:**
		
		•	Mentions - Number of posts related to the coin in tracked social media sources in the requested date with TimeFrame required.
		•	MentionsChange - Percentage change of Mentions in the requested date vs the average of the same metric in the past 14 days.
		•	Sentiment - Democratic Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative)in the requested date with TimeFrame required.  [Democratic Sentiment is a sentiment that gives the same weight to each mention related to the coin despite the number of followers for each mention]
		•	SentimentChange - Percentage change of Democratic Sentiment in the requested date vs the average of the same metric in the past 14 days.
		•	WeightedSentiment –  Weighted Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative)  in the requested date with TimeFrame required. [ Weighted Sentiment is a sentiment that gives more weight to mentions with more followers] 
		•	WeightedSentimentChange - Percentage change of Weighted Sentiment in the requested date vs the average of the same metric in the past 14 days.
		•	Followers - Sum of followers reached by coin related mentions in the requested date with TimeFrame required.
		•	FollowersChange - Percentage change of Followers Reach in the requested date vs the average of the same metric in the past 14 days.
		
		**Details:**
		
		- All dates are in UTC
		- Historical date is only avaliable from 7 days back"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/tokenhistmetrics/gethistoricalcoinmetricsbyname"
    querystring = {'securityName': securityname, 'date': date, }
    if timeframe:
        querystring['timeFrame'] = timeframe
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_historical_metrics_by_ticker(securityticker: str, date: str, timeframe: str='1D', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve  coin´s historical social media metrics by Coin ticker
		In the case you want to retrieve real time data check "Get Coin Metrics by Name" or "Get Coin Metrics by Ticker"
		
		**Input:** 
		-  Date (MM/DD/YYYY HH:HH AM/PM) (Required)
		- Security Ticker: Coin Ticker (Required)
		- TimeFrame: 1d, 1h, 8h (Optional)
		
		**Output:**
		
		- Sentiment
		- Sentiment change
		- Weighted sentiment
		- Weighted sentiment change
		- Followers
		- Followers change
		- Mentions
		- Mentions change
		- Security ticker
		- Security name
		- TimeFrame
		- DataTimeStamp
		
		**Definitions:**
		
		•	Mentions - Number of posts related to the coin in tracked social media sources in the requested date with TimeFrame required.
		•	MentionsChange - Percentage change of Mentions in the requested date vs the average of the same metric in the past 14 days.
		•	Sentiment - Democratic Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative)in the requested date with TimeFrame required.  [Democratic Sentiment is a sentiment that gives the same weight to each mention related to the coin despite the number of followers for each mention]
		•	SentimentChange - Percentage change of Democratic Sentiment in the requested date vs the average of the same metric in the past 14 days.
		•	WeightedSentiment –  Weighted Sentiment Score out of 100 Points (100 for the most positive, 50 for neutral and 0 for the most negative)  in the requested date with TimeFrame required. [ Weighted Sentiment is a sentiment that gives more weight to mentions with more followers] 
		•	WeightedSentimentChange - Percentage change of Weighted Sentiment in the requested date vs the average of the same metric in the past 14 days.
		•	Followers - Sum of followers reached by coin related mentions in the requested date with TimeFrame required.
		•	FollowersChange - Percentage change of Followers Reach in the requested date vs the average of the same metric in the past 14 days.
		
		**Details:**
		
		- All dates are in UTC
		- Historical date is only avaliable from 7 days back"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/tokenhistmetrics/gethistoricalcoinmetricsbyticker"
    querystring = {'securityTicker': securityticker, 'date': date, }
    if timeframe:
        querystring['timeFrame'] = timeframe
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_supported_coins(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the supported coins, the result contains pairs of Name and Ticker of each coin ordered alphabetically"
    
    """
    url = f"https://crowdsense1.p.rapidapi.com/api/tokenmetrics/getsupportedcoins"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdsense1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

