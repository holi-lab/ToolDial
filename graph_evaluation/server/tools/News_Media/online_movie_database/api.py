import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def title_get_crazycredits(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get crazycredits in specific title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-crazycredits"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_all_images(tconst: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available photos of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        limit: Default is 3000, up to the value of totalImageCount
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-all-images"
    querystring = {'tconst': tconst, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_images(tconst: str, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get photo shots of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        limit: Up to 100
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-images"
    querystring = {'tconst': tconst, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_best_picture_winners(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get best picture winners"
    
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-best-picture-winners"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_reviews(tconst: str, currentcountry: str='US', purchasecountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        purchasecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-reviews"
    querystring = {'tconst': tconst, }
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    if purchasecountry:
        querystring['purchaseCountry'] = purchasecountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_list_popular_genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List popular genres"
    
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/list-popular-genres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_news(tconst: str, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news related to the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        limit: Up to 100


        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-news"
    querystring = {'tconst': tconst, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_release_expectation_bundle(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get release expectation of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-release-expectation-bundle"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_v2_get_business(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get business (Box Office) information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/v2/get-business"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_top_rated_tv_shows(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top rated 250 tv shows"
    
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-top-rated-tv-shows"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_base(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get base information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-base"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_v2_get_popular_movies_by_genre(genre: str, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get popular movies by genre"
    genre: One of the following : action|adventure|animation|biography|comedy|crime|documentary|drama|family|fantasy|film-noir|game-show||history|horror|music|musical|mystery|news|reality-tv|romance|sci-fi|short|sport|talk-show|thriller|war|western
        limit: Number of items per response
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/v2/get-popular-movies-by-genre"
    querystring = {'genre': genre, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_top_crew(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top crew information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-top-crew"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_plots(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get plots information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-plots"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_technical(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get technical information"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-technical"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_coming_soon_tv_shows(today: str=None, currentcountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get coming soon TV shows"
    today: Date format as following yyyy-MM-dd. Ex : 2020-07-27
        currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-coming-soon-tv-shows"
    querystring = {}
    if today:
        querystring['today'] = today
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_most_popular_tv_shows(homecountry: str='US', purchasecountry: str='US', currentcountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get most popular TV shows"
    homecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        purchasecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-most-popular-tv-shows"
    querystring = {}
    if homecountry:
        querystring['homeCountry'] = homecountry
    if purchasecountry:
        querystring['purchaseCountry'] = purchasecountry
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_versions(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get versions information that supported"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-versions"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_overview_details(tconst: str, currentcountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get overview information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-overview-details"
    querystring = {'tconst': tconst, }
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_full_credits(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full list of casts and crews relating to specific title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-full-credits"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_top_stripe(tconst: str, purchasecountry: str='US', currentcountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top stripe of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        purchasecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-top-stripe"
    querystring = {'tconst': tconst, }
    if purchasecountry:
        querystring['purchaseCountry'] = purchasecountry
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_top_cast(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get id list of top cast to pass into /title/get-charname-list endpoint"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-top-cast"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_details(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-details"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_v2_find(title: str, userratingmin: int=None, sortarg: str='moviemeter,asc', watchoption: str=None, limit: int=20, genre: str=None, releasedatemin: str=None, paginationkey: int=None, titletype: str=None, releasedatemax: str=None, runtimemin: int=None, numvotesmin: int=None, keyword: str=None, group: str=None, runtimemax: int=None, primarylanguage: str=None, primarycountry: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find title by whatever you are familiar with, such as : name of title, album, song, etc… with options and filters"
    title: Anything that you are familiar with, such as : name of title, album, song, etc...
        userratingmin: One of the following : 6.0|7.0|8.0|9.0
        sortarg: One of the following : moviemeter,asc|moviemeter,desc|alpha,asc|alpha,desc|user&#95;rating,asc|user&#95;rating,desc|num&#95;votes,asc|num&#95;votes,desc|boxoffice&#95;gross&#95;us,asc|boxoffice&#95;gross&#95;us,desc|runtime,asc|runtime,desc|year,asc|year,desc|release&#95;date,asc|release&#95;date,desc
        watchoption: One of the following :  has&#95;video&#95;prime&#95;instant&#95;video&#95;us|has&#95;video&#95;amazon&#95;instant&#95;video&#95;us|has&#95;video&#95;freedive&#95;us. Separated by comma for multiple options. Ex : has&#95;video&#95;prime&#95;instant&#95;video&#95;us,has&#95;video&#95;freedive&#95;us,...
        limit: Number of items per response, for paging purpose. Maximum is 20
        genre: One of the following : comedy|horror|romance|thriller|sci-fi|drama|action|adventure|animation|biography|crime|documentary|family|fantasy|film-noir|game-show|history|music|musical|mystery|news|reality-tv|sport|talk-show|war|western. Separated by comma for multiple options. Ex : comedy,adventure,...
        releasedatemin: The format is yyyy-MM-dd. Ex : 2002-01-01
        paginationkey: The page index, for paging purpose.
        titletype: One of the following : movie|tvSeries|short|tvEpisode|tvMiniSeries|tvMovie|tvSpecial|tvShort|videoGame|video|musicVideo|podcastSeries|podcastEpisode. Separated by comma for multiple options. Ex : movie,tvEpisode,...
        releasedatemax: The format is yyyy-MM-dd. Ex : 2023-12-31
        runtimemin: In minutes. Ex : 60
        numvotesmin: One of the following : 1000|10000|100000|1000000
        keyword: One of the following : action hero,alien invasion,alternate history,anime,avant garde,b-movie,bank robbery,based on comic book,based on novel,based on play,based on true story,bechdel test passed,black comedy,chick flick,coming of age,criminal mastermind,cult film,dark fantasy,dc comics,dystopia,epic,experimental film,f rated,femme fatale,good versus evil,haunting,heist,high school,independent film,kidnapping,kung fu,magical realism,marvel comics,mockumentary,one man army,parallel world,paranormal,post apocalypse,sequel,space travel,spaghetti western,spoof,steampunk,superhero,supernatural,time travel,triple f rated,vampire,zombie. Separated by comma for multiple options. Ex : action hero,alternate history,...
        group: One of the following : best&#95;director&#95;winner,best&#95;picture&#95;winner,oscar&#95;winner,globes&#95;winner,emmys&#95;winner,oscar&#95;nominee,globes&#95;nominee,emmys&#95;nomiee. Separated by comma for multiple options. Ex : best&#95;director&#95;winner,oscar&#95;winner,...
        runtimemax: In minutes. Ex : 180
        primarylanguage: One of the following : ab|qac|guq|qam|af|qas|ak|sq|alg|ase|am|grc|apa|ar|an|arc|arp|hy|as|aii|ast|ath|asf|awa|ay|az|qbd|ban|bm|bn|eu|bsc|be|bem|ber|bho|qbi|qbh|bs|bzs|br|bfi|bg|my|yue|ca|ccp|qax|ce|chr|chy|hne|zh|kw|co|cr|mus|qal|crp|hr|cro|cs|da|prs|dso|din|qaw|doi|nl|dyu|dz|qbc|frs|egy|en|eo|et|ee|qbg|fo|fil|fi|qbn|fon|fr|fsl|ff|fvr|gl|ka|de|gsg|grb|el|gn|gu|gnn|gup|hai|ht|hak|bgc|qav|ha|haw|he|hi|hmn|qab|hop|hu|iba|qag|is|icl|ins|id|iu|ik|ga|iru|it|ja|jsl|dyo|ktz|qbf|kea|kab|kl|xal|kn|kpj|mjw|kar|kk|kca|kha|km|ki|rw|qar|tlh|kfa|kok|ko|kvk|khe|qaq|kro|kyw|qbb|ku|kwk|ky|lbj|lad|lo|la|lv|lif|ln|lt|nds|lb|mk|qbm|mag|mai|mg|ms|ml|pqm|qap|mt|mnc|cmn|man|mni|mi|arn|mr|mh|mas|mls|myn|men|mic|enm|nan|min|mwl|qmt|lus|moh|mn|moe|qaf|mfe|qbl|nah|qba|nv|nbf|nap|yrk|ne|ncg|zxx|nai|nd|no|qbk|nyk|ny|oc|or|oj|qaz|ang|non|pap|qaj|ps|paw|fa|qai|pl|qah|pt|fuf|pa|tsz|qu|qya|raj|qbj|xrr|ro|rm|rom|rtm|ru|rsl|qao|qae|sah|sm|sa|sc|qay|gd|sr|qbo|srr|qad|qau|sn|shh|scn|sjn|sd|si|sio|sk|sl|so|son|snk|wen|qbe|st|es|ssp|srn|sw|sv|gsw|syl|tl|tg|tmh|ta|tac|tt|te|qak|th|bo|qan|tli|tpi|to|ts|tsc|tn|tcy|tup|tr|tk|tyv|tzo|uk|ukl|qat|ur|uz|vi|qaa|was|cy|wo|xh|yap|yi|yo|zu. Separated by comma for multiple options. Ex : en,fr,...
        primarycountry: One of the following : af|ax|al|dz|as|ad|ao|ai|aq|ag|ar|am|aw|au|at|az|bs|bh|bd|bb|by|be|bz|bj|bm|bt|bo|ba|bw|bv|br|io|vg|bn|bg|bf|bumm|bi|kh|cm|ca|cv|bq|ky|cf|td|cl|cn|cx|cc|co|km|cg|cd|ck|cr|ci|hr|cu|an|cy|cz|cshh|dk|dj|dm|do|ddde|ec|eg|sv|gq|er|ee|et|fk|fo|yucs|fj|fi|fr|gf|pf|tf|ga|gm|ge|de|gh|gi|gr|gl|gd|gp|gu|gt|gg|gn|gw|gy|ht|hm|hn|hk|hu|is|in|id|ir|iq|ie|im|il|it|jm|jp|je|jo|kz|ke|xyu|ki|xko|xkv|kw|kg|la|lv|lb|ls|lr|ly|li|lt|lu|mo|mk|mg|mw|my|mv|ml|mt|mh|mq|mr|mu|yt|mx|fm|md|mc|mn|me|ms|ma|mz|mm|na|nr|np|nl|nc|nz|ni|ne|ng|nu|nf|kp|vdvn|mp|no|om|pk|pw|xpi|pa|pg|py|pe|ph|pn|pl|pt|pr|qa|re|ro|ru|rw|ws|sm|st|sa|sn|rs|csxx|sc|xsi|sl|sg|sk|si|sb|so|za|gs|kr|suhh|es|lk|bl|sh|kn|lc|mf|pm|vc|sd|sr|sj|sz|se|ch|sy|tw|tj|tz|th|tl|tg|tk|to|tt|tn|tr|tm|tc|tv|um|vi|ug|ua|ae|gb|us|uy|uz|vu|va|ve|vn|wf|xwg|eh|ye|zrcd|zm|zw. Separated by comma for multiple options. Ex : us,fr,...
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/v2/find"
    querystring = {'title': title, }
    if userratingmin:
        querystring['userRatingMin'] = userratingmin
    if sortarg:
        querystring['sortArg'] = sortarg
    if watchoption:
        querystring['watchOption'] = watchoption
    if limit:
        querystring['limit'] = limit
    if genre:
        querystring['genre'] = genre
    if releasedatemin:
        querystring['releaseDateMin'] = releasedatemin
    if paginationkey:
        querystring['paginationKey'] = paginationkey
    if titletype:
        querystring['titleType'] = titletype
    if releasedatemax:
        querystring['releaseDateMax'] = releasedatemax
    if runtimemin:
        querystring['runtimeMin'] = runtimemin
    if numvotesmin:
        querystring['numVotesMin'] = numvotesmin
    if keyword:
        querystring['keyword'] = keyword
    if group:
        querystring['group'] = group
    if runtimemax:
        querystring['runtimeMax'] = runtimemax
    if primarylanguage:
        querystring['primaryLanguage'] = primarylanguage
    if primarycountry:
        querystring['primaryCountry'] = primarycountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_videos(tconst: str, region: str='US', limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get promoted, trailer video clips"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        region: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        limit: Up to 200
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-videos"
    querystring = {'tconst': tconst, }
    if region:
        querystring['region'] = region
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_find(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find title by whatever you are familiar with, such as : name of title, album, song, etc…"
    q: Anything that you are familiar with, such as : name of title, album, song, etc...
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/find"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_company_credits(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get company credits"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-company-credits"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_business_deprecated(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get business (Box Office) information of the title
		* This endpoint is deprecated. You need to use .../title/v2/get-business endpoint instead"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-business"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_goofs(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get goofs in specific title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-goofs"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_sound_tracks(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sound tracks in specific title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-sound-tracks"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_quotes(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get quotes in specific title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-quotes"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_trivia(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trivia of title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-trivia"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_releases(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get release information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-releases"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_meta_data(ids: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief information of movies by their id"
    ids: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947. Pass this parameter multiple times (up to 50) for multiple movies, such as : ids=tt4154756&ids=tt0050825&…
        region: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-meta-data"
    querystring = {'ids': ids, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_filming_locations(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get filming locations information"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-filming-locations"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_more_like_this(tconst: str, currentcountry: str='US', purchasecountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similar title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        purchasecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-more-like-this"
    querystring = {'tconst': tconst, }
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    if purchasecountry:
        querystring['purchaseCountry'] = purchasecountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_charname_list(tconst: str, is_id: str, marketplace: str='ATVPDKIKX0DER', purchasecountry: str='US', currentcountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get characters information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        id: You need to extract the value started with 'nm' returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667. Pass this parameter multiple times to get one or more characters, such as : &id=nm3964231&id=nm3964232&…

        marketplace: Look for value of amazonProductId field returned by title/get-top-stripe endpoint
        purchasecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-charname-list"
    querystring = {'tconst': tconst, 'id': is_id, }
    if marketplace:
        querystring['marketplace'] = marketplace
    if purchasecountry:
        querystring['purchaseCountry'] = purchasecountry
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_genres(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get genres information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-genres"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_synopses(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get synopses information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-synopses"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_production_status(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get production status of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-production-status"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_top_rated_movies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top rated 250 movies"
    
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-top-rated-movies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_video_playback(viconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get promoted, trailer video clips"
    viconst: You need to extract for the value started with 'vi' of videos/id JSON object returned in …/title/get-videos endpoint. Ex : vi589675545
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-video-playback"
    querystring = {'viconst': viconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_seasons(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all seasons series information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-seasons"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_awards_summary(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get awards summary related to the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-awards-summary"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_user_reviews(tconst: str, paginationkey: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user reviews of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        paginationkey: The value of paginationKey field that returned right in this endpoint
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-user-reviews"
    querystring = {'tconst': tconst, }
    if paginationkey:
        querystring['paginationKey'] = paginationkey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_hero_with_promoted_video(tconst: str, purchasecountry: str='US', currentcountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get heroes information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        purchasecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-hero-with-promoted-video"
    querystring = {'tconst': tconst, }
    if purchasecountry:
        querystring['purchaseCountry'] = purchasecountry
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_awards(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get awards related to the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-awards"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_metacritic(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get metacritic information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-metacritic"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_ratings(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ratings of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-ratings"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_coming_soon_movies(today: str=None, purchasecountry: str='US', homecountry: str='US', currentcountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get coming soon movies"
    today: Date format as following yyyy-MM-dd. Ex : 2020-07-27
        purchasecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        homecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-coming-soon-movies"
    querystring = {}
    if today:
        querystring['today'] = today
    if purchasecountry:
        querystring['purchaseCountry'] = purchasecountry
    if homecountry:
        querystring['homeCountry'] = homecountry
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_taglines(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get taglines information of the title"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-taglines"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_parental_guide(tconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get parent guide information for specific movie"
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-parental-guide"
    querystring = {'tconst': tconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_get_most_popular_movies(purchasecountry: str='US', currentcountry: str='US', homecountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get most popular movies"
    purchasecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        homecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/title/get-most-popular-movies"
    querystring = {}
    if purchasecountry:
        querystring['purchaseCountry'] = purchasecountry
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    if homecountry:
        querystring['homeCountry'] = homecountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_get_awards(nconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get awards of actor or actress"
    nconst: You need to extract the value started with "nm" returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/get-awards"
    querystring = {'nconst': nconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_list_most_popular_celebs(currentcountry: str='US', purchasecountry: str='US', homecountry: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get most popular movies"
    currentcountry: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        purchasecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        homecountry: Accept one of following US|GB|DE|IN|IT|FR|JP|CA|ES
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/list-most-popular-celebs"
    querystring = {}
    if currentcountry:
        querystring['currentCountry'] = currentcountry
    if purchasecountry:
        querystring['purchaseCountry'] = purchasecountry
    if homecountry:
        querystring['homeCountry'] = homecountry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_list_born_today(month: int, day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all actors and actresses by day and month"
    month: The month of birth of actors
        day: The day of birth of actors
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/list-born-today"
    querystring = {'month': month, 'day': day, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_get_filmography_appearances(tconst: str, nconst: str, category: str='actor', region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get filmography appearances of an actor, actress, etc..."
    tconst: You need to extract the value started with 'tt' of id field returned from …/title/auto-complete or …/title/find endpoint. Ex : tt0944947
        nconst: You need to extract the value started with 'nm' returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667
        category: Accept one of following actor|actress|soundtrack|director|writer
        region: Accept one of following AR|AU|CA|CL|ES|DE|IT|MX|NZ|PT|ES|GB|US
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/get-filmography-appearances"
    querystring = {'tconst': tconst, 'nconst': nconst, }
    if category:
        querystring['category'] = category
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_get_known_for(nconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get known-for of actor or actress"
    nconst: You need to extract the value started with "nm" returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/get-known-for"
    querystring = {'nconst': nconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_get_interesting_jobs(nconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get interesting jobs of actor or actress"
    nconst: You need to extract the value started with \\\"nm\\\" returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/get-interesting-jobs"
    querystring = {'nconst': nconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_get_awards_summary(nconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get awards summary of actor or actress"
    nconst: You need to extract the value started with "nm" returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/get-awards-summary"
    querystring = {'nconst': nconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_get_all_videos(nconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all videos of actor or actress"
    nconst: You need to extract the value started with "nm" returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/get-all-videos"
    querystring = {'nconst': nconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_get_all_news(nconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all news of actor or actress"
    nconst: You need to extract the value started with "nm" returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/get-all-news"
    querystring = {'nconst': nconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_get_all_images(nconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all images of actor or actress"
    nconst: You need to extract the value started with "nm" returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/get-all-images"
    querystring = {'nconst': nconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_get_all_filmography(nconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all filmography of actor or actress"
    nconst: You need to extract the value started with "nm" returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/get-all-filmography"
    querystring = {'nconst': nconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_get_bio(nconst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get biography of actor or actress"
    nconst: You need to extract the value started with "nm" returned in …/actors/list-born-today or …/actors/list-most-popular-celebs endpoint. Ex : nm0001667
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/actors/get-bio"
    querystring = {'nconst': nconst, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestion by term or phrase"
    q: Anything that you are familiar with, such as : name of title, album, song, etc...
        
    """
    url = f"https://online-movie-database.p.rapidapi.com/auto-complete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

