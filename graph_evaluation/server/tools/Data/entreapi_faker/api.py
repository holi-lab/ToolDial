import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def longitute(max: int=None, min: int=None, precision: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random longitude."
    max: Maximum value for latitude.
        min: Minimum value for latitude.
        precision: Precision for latitude.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/longitude"
    querystring = {}
    if max:
        querystring['max'] = max
    if min:
        querystring['min'] = min
    if precision:
        querystring['precision'] = precision
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def boolean(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a boolean value."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/datatype/boolean"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def past(refdate: str=None, years: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a date value in the past."
    refdate: Starting reference date
        years: Number of years for the range of dates.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/date/past"
    querystring = {}
    if refdate:
        querystring['refDate'] = refdate
    if years:
        querystring['years'] = years
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_url(width: int=None, height: int=None, userandomize: bool=None, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an image url."
    width: Width of the image. Default is 640.
        height: Height of the image. Default is 480.
        userandomize: Add a random number parameter to the returned url.
        category: The category for the image.  Can be one:
abstract,
animal,
avatar,
business,
cats,
city,
fashion,
food, 
nature,
nightlife,
people,
sports,
technics,
transport
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/imageUrl"
    querystring = {}
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if userandomize:
        querystring['useRandomize'] = userandomize
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sentence(wordcount: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a sentence of lorem ipsum."
    wordcount: Number of words in the sentence.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/lorem/sentence"
    querystring = {}
    if wordcount:
        querystring['wordCount'] = wordcount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gender(usebinary: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly select a gender."
    usebinary: Use binary genders only.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/gender"
    querystring = {}
    if usebinary:
        querystring['useBinary'] = usebinary
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prefix(gender: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a prefix (ie Mr., Mrs., etc)"
    gender: Optional gender.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/prefix"
    querystring = {}
    if gender:
        querystring['gender'] = gender
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def number_value(min: int=None, max: int=None, precision: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a number value."
    min: Minimu value.
        max: Maximum value.
        precision: Precision of the number.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/random/number"
    querystring = {}
    if min:
        querystring['min'] = min
    if max:
        querystring['max'] = max
    if precision:
        querystring['precision'] = precision
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def url(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a url."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/url"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def avatar_image(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an avatar."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/avatar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image(width: int=None, height: int=None, userandomize: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an image of any type."
    width: Width of the image. Default is 640.
        height: Height of the image. Default is 480.
        userandomize: Add a random number parameter to the returned url.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/image"
    querystring = {}
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if userandomize:
        querystring['useRandomize'] = userandomize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def food(height: int=None, userandomize: bool=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a food image."
    height: Height of the image. Default is 480.
        userandomize: Add a random number parameter to the returned url.
        width: Width of the image. Default is 640.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/food"
    querystring = {}
    if height:
        querystring['height'] = height
    if userandomize:
        querystring['useRandomize'] = userandomize
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fashion(userandomize: bool=None, height: int=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a fashion image."
    userandomize: Add a random number parameter to the returned url.
        height: Height of the image. Default is 480.
        width: Width of the image. Default is 640.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/fashion"
    querystring = {}
    if userandomize:
        querystring['useRandomize'] = userandomize
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datauri(height: int, width: int, color: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a dataUri square of the color specified."
    height: Height of the image. Default is 480.
        width: Width of the image. Default is 640.
        color: What color to use.  Default is grey.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/dataUri"
    querystring = {'height': height, 'width': width, }
    if color:
        querystring['color'] = color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities(userandomize: bool=None, width: int=None, height: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a city image."
    userandomize: Add a random number parameter to the returned url.
        width: Width of the image. Default is 640.
        height: Height of the image. Default is 480.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/city"
    querystring = {}
    if userandomize:
        querystring['useRandomize'] = userandomize
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cats(userandomize: bool=None, height: int=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a cat image."
    userandomize: Add a random number parameter to the returned url.
        height: Height of the image. Default is 480.
        width: Width of the image. Default is 640.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/cats"
    querystring = {}
    if userandomize:
        querystring['useRandomize'] = userandomize
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business(width: int=None, height: int=None, userandomize: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a business image."
    width: Width of the image. Default is 640.
        height: Height of the image. Default is 480.
        userandomize: Add a random number parameter to the returned url.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/business"
    querystring = {}
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if userandomize:
        querystring['useRandomize'] = userandomize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def avatar(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an avatar image."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/avatar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def animals(height: int=None, width: int=None, userandomize: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an animals image."
    height: Height of the image. Default is 480.
        width: Width of the image. Default is 640.
        userandomize: Add a random number parameter to the returned url.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/animals"
    querystring = {}
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    if userandomize:
        querystring['useRandomize'] = userandomize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def abstract(userandomize: bool=None, width: int=None, height: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an abstract image."
    userandomize: Add a random number parameter to the returned url.
        width: Width of the image. Default is 640.
        height: Height of the image. Default is 480.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/abstract"
    querystring = {}
    if userandomize:
        querystring['useRandomize'] = userandomize
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contextual_card(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate contextual card data."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/contextualCard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_card(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate user card data."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/userCard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def slugify(string: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Slugify a specified string."
    string: The string to slugify.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/slugify"
    querystring = {}
    if string:
        querystring['string'] = string
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def replace_symbols(string: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Replace symbols in a specified string with letters and/or numbers (# replaces number, ? replaces letter, * replaces either)."
    string: A string mask containing symbols to replace.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/replaceSymbols"
    querystring = {}
    if string:
        querystring['string'] = string
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def replace_symbol_with_number(symbol: str=None, string: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Replace the specified symbol in the specified string with random numbers."
    symbol: The symbol to replace with number.  Default is #.
        string: The specified mask to use (ie ###-##-####).  Default is empty string.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/replaceSymbolWithNumber"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if string:
        querystring['string'] = string
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def replace_credit_card_symbols(string: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Replace credit card symbols with random numbers."
    string: The string template (ie 4772-####-####-####-###). Default is 6453-####-####-####-###L.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/replaceCreditCardSymbols"
    querystring = {}
    if string:
        querystring['string'] = string
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def repeat_string(string: str, num: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Repeat the specified string a number of times."
    string: The string to repeat.
        num: The number of times to repeat the string.  Default is 0.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/repeatString"
    querystring = {'string': string, }
    if num:
        querystring['num'] = num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regexp_style_string_parse(string: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Expand a regex express with random data."
    string: The regex expression to expand.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/regexpStyleStringParse"
    querystring = {}
    if string:
        querystring['string'] = string
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mustache(data: str, string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Expand a mustache express with specified data."
    data: The data to use for the mustache expression.
        string: The mustache expression.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/mustache"
    querystring = {'data': data, 'string': string, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_transaction(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a transaction."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/createTransaction"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_card(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate card data."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/helpers/createCard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verb(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a hacker related verb."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/hacker/verb"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phrase(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a hacker related phrase."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/hacker/phrase"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def noun(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a hacker related noun."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/hacker/noun"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ingverb(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a hacker related verb ending in -ing."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/hacker/ingverb"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adjective(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a hacker related adjective."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/hacker/adjective"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def abbreviation(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a hacker related abbreviation."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/hacker/abbreviation"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def short_sha(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a GIT short hash."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/git/shortSha"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commit_sha(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a GIT commit hash."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/git/commitSha"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commit_message(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a GIT commit message."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/git/commitMessage"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commit_entry(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a GIT commit entry."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/git/commitEntry"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def branch(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a sample GIT branch name."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/git/branch"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transaction_type(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a transaction type."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/transactionType"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transaction_description(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a transaction description."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/transactionDescription"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def routing_number(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a routing number."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/routingNumber"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mask(length: int=None, parens: bool=None, ellipsis: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a mask."
    length: Length of the mask.
        parens: Use parenthesis in the mask.
        ellipsis: Use ellipsis in the mask.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/mask"
    querystring = {}
    if length:
        querystring['length'] = length
    if parens:
        querystring['parens'] = parens
    if ellipsis:
        querystring['ellipsis'] = ellipsis
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def litecoin_address(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a litecoin address."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/litecoinAddress"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iban(useformatted: bool=None, countrycode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an iban."
    useformatted: Use formatted.
        countrycode: Country code to use.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/iban"
    querystring = {}
    if useformatted:
        querystring['useformatted'] = useformatted
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ethereum_address(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an ethereum address."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/ethereumAddress"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_symbol(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a currency symbol."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/currencySymbol"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a currency name."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/currencyName"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_code(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a currency code."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/currencyCode"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def credit_card_number(provider: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a credit card number."
    provider: The card provider.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/creditCardNumber"
    querystring = {}
    if provider:
        querystring['provider'] = provider
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def credit_card_cvv(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a credit card CCV."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/creditCardCVV"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bitcoin_address(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a bitcoin address."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/bitcoinAddress"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bic(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate BIC"
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/bic"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amount(symbol: str=None, max: int=None, min: int=None, useautoformat: bool=None, dec: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an amount."
    symbol: Currency symbol.
        max: Maximum value for amount.
        min: Minimum value for amount.
        useautoformat: Use auto format.
        dec: Number of decimal places.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/amount"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if max:
        querystring['max'] = max
    if min:
        querystring['min'] = min
    if useautoformat:
        querystring['useAutoFormat'] = useautoformat
    if dec:
        querystring['dec'] = dec
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an account name."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/accountName"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account(length: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an account number."
    length: The length of the account number.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/finance/account"
    querystring = {}
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekday(context: str=None, abbr: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a weekday."
    context: Use an alternate context.
        abbr: Generate an abbreviation for the weekday.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/date/weekday"
    querystring = {}
    if context:
        querystring['context'] = context
    if abbr:
        querystring['abbr'] = abbr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soon(days: int=None, refdate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a soon date value."
    days: Number of days for the range of dates.
        refdate: Starting reference date
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/date/soon"
    querystring = {}
    if days:
        querystring['days'] = days
    if refdate:
        querystring['refDate'] = refdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent(refdate: str=None, days: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a recent date value."
    refdate: Starting reference date
        days: Number of days for the range of dates.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/date/recent"
    querystring = {}
    if refdate:
        querystring['refDate'] = refdate
    if days:
        querystring['days'] = days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def month(abbr: bool=None, context: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a month."
    abbr: Generate an abbreviation for the month.
        context: Use an alternate context.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/date/month"
    querystring = {}
    if abbr:
        querystring['abbr'] = abbr
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def future(years: int=None, refdate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a date value in the future."
    years: Number of years for the range of dates.
        refdate: Starting reference date
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/date/future"
    querystring = {}
    if years:
        querystring['years'] = years
    if refdate:
        querystring['refDate'] = refdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def between(to: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random date within a date range."
    to: End of the date range.
        from: Star of the date range.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/date/between"
    querystring = {'to': to, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def betweens(to: str, is_from: str, num: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate dates value within a date range."
    to: End of date range.
        from: Start of date range.
        num: Number of values to create (default is 3)
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/date/betweens"
    querystring = {'to': to, 'from': is_from, }
    if num:
        querystring['num'] = num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uuid(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a uuid value."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/datatype/uuid"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def string(length: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a string value."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/datatype/string"
    querystring = {}
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def number(precision: int=None, min: int=None, max: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a number value."
    precision: Precision of the number.
        min: Minimum numer value.
        max: Maximum number value.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/datatype/number"
    querystring = {}
    if precision:
        querystring['precision'] = precision
    if min:
        querystring['min'] = min
    if max:
        querystring['max'] = max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def json(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a json object value."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/datatype/json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hexadecimal(count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a hexadecimal value."
    count: Count of hexadecimal digits.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/datatype/hexaDecimal"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def float(max: int=None, precision: int=None, min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a float value."
    max: Maximum float value.
        precision: Precision for the float value.
        min: Minimum float value.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/datatype/float"
    querystring = {}
    if max:
        querystring['max'] = max
    if precision:
        querystring['precision'] = precision
    if min:
        querystring['min'] = min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datetime(max: int=None, min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a datetime value."
    max: Maximum datetime value.
        min: Minimum datetime value.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/datatype/datetime"
    querystring = {}
    if max:
        querystring['max'] = max
    if min:
        querystring['min'] = min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def array(length: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an array of data."
    length: The length of the array.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/datatype/array"
    querystring = {}
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def column_type(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a column type."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/database/type"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def engine(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a database engine name."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/database/engine"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def column_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a column name."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/database/column"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def collation(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a database collation."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/database/collation"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def suffixes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the possible suffixes."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/suffixes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_suffix(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a company suffix."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/companySuffix"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_name(format: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "randomly generate a company name."
    format: The format for the company name. Values:
0, 1, or 2
where:
0 = {{name.lastName}} {{company.companySuffix}}
1 = {{name.lastName}} - {{name.lastName}}
2 = {{name.lastName}}, {{name.lastName}} and {{name.lastName}}

        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/companyName"
    querystring = {'format': format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def catch_phrase_noun(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a company catch phrase noun."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/catchPhraseNoun"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def catch_phrase_descriptor(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a company catch phrase descriptor."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/catchPhraseDescriptor"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def catch_phrase_adjective(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a company catch phrase adjective."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/catchPhraseAdjective"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def catch_phrase(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate company catch phrase."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/catchPhrase"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bs_noun(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate company BS noun."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/bsNoun"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bs_buzz(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate company BS buzz."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/bsBuzz"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bs_adjective(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate company BS adjective."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/bsAdjective"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate company BS."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/company/bs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a product name."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/commerce/productName"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_material(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a product material."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/commerce/productMaterial"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_description(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a product description."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/commerce/productDescription"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_adjective(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a product adjective."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/commerce/productAdjective"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a product."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/commerce/product"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price(max: int=1000, dec: int=2, min: int=10, symbol: str='$', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a price."
    symbol: Currency symbol.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/commerce/price"
    querystring = {}
    if max:
        querystring['max'] = max
    if dec:
        querystring['dec'] = dec
    if min:
        querystring['min'] = min
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def department(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a department."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/commerce/department"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def color(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a color."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/commerce/color"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def animal_type(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an animal type."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/type"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def snake(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a snake species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/snake"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rabbit(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a rabbit species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/rabbit"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lion(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a lion species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/lion"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insect(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an insect species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/insect"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def horse(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a horse species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/horse"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fish(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a fish species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/fish"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dog(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a dog species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/dog"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crocodilia(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a crocodilia species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/crocodilia"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cow(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a cow species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/cow"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cetacean(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a cetacean species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/cetacean"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cat(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a cat species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/cat"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bird(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a bird species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/bird"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vrm(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a vehicle registration mark."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/vehicle/vrm"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vin(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a vehicle identification number."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/vehicle/vin"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vehicle(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly select a vehicle."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/vehicle/vehicle"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly select a vehicle type."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/vehicle/type"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bicycle(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly select a bicycle type."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/vehicle/bicycle"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def model(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly select a model."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/vehicle/model"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manufacturer(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly select a manufacturer."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/vehicle/manufacturer"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuel(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly select a fuel type."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/vehicle/fuel"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def color_value(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly select a color."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/vehicle/color"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_time(outputtype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a recent timestamp."
    outputtype: Output type can be unix, abbr, or wide.  Default is unix.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/time/recent"
    querystring = {}
    if outputtype:
        querystring['outputType'] = outputtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def common_file_name(ext: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a common file name."
    ext: Extension to use.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/system/commonFileName"
    querystring = {}
    if ext:
        querystring['ext'] = ext
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def semver(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a semver."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/system/semver"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mime_type(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a mime type."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/system/mimeType"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def file_type(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a file type."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/system/fileType"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def file_path(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a file path."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/system/filePath"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def file_ext(mimetype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a file extension. mimeType"
    mimetype: The mime type to use.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/system/fileExt"
    querystring = {}
    if mimetype:
        querystring['mimeType'] = mimetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def directory_path(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a directory path."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/system/directoryPath"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def common_file_type(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a common file type."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/system/commonFileType"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def common_file_ext(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a common file extension."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/system/commonFileExt"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def word_values(count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a list of words values."
    count: Number of words to return.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/random/words"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def word_value(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random word value."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/random/word"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uuid_value(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a uuid value."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/random/uuid"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def object_element(field: str=None, object: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly select an object element."
    field: Return field name instead of value.  Valid value is "key".
        object: The object that has elements.  Default is { "foo": "bar", "too": "car" }.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/random/objectElement"
    querystring = {}
    if field:
        querystring['field'] = field
    if object:
        querystring['object'] = object
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locale(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a locale."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/random/locale"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_image(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an image url."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/random/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hexadecimal_value(count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a hexadecimal value."
    count: Number of characters to return.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/random/hexaDecimal"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def float_value(max: int=None, precision: int=None, min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a float value."
    max: The maximum value.
        precision: The precision.
        min: The minimum value.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/random/float"
    querystring = {}
    if max:
        querystring['max'] = max
    if precision:
        querystring['precision'] = precision
    if min:
        querystring['min'] = min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def boolean_value(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a boolean value."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/random/boolean"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phone_number_format(phoneformatsarrayindex: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate phone number from format index."
    phoneformatsarrayindex: The index of which format to use. Default is 0.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/phone/phoneNumberFormat"
    querystring = {}
    if phoneformatsarrayindex:
        querystring['phoneFormatsArrayIndex'] = phoneformatsarrayindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phone_number(format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a phone number."
    format: The phone number format to use.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/phone/phoneNumber"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phone_formats(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a phone format string."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/phone/phoneFormats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a title."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/title"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def suffix(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a suffix (ie Jr, II, etc)"
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/suffix"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def middle_name(gender: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a middle name."
    gender: Optional gender.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/middleName"
    querystring = {}
    if gender:
        querystring['gender'] = gender
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_name(gender: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a last name."
    gender: Optional gender.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/lastName"
    querystring = {}
    if gender:
        querystring['gender'] = gender
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def job_type(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a job type."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/jobType"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def job_title(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a job title."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/jobTitle"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def job_descriptor(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a job descriptor."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/jobDescriptor"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def job_area(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a job area."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/jobArea"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def first_name(gender: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a first name."
    gender: Optional gender.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/firstName"
    querystring = {}
    if gender:
        querystring['gender'] = gender
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_name(lastname: str=None, gender: str=None, firstname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a name."
    lastname: Optional last name.
        gender: Optional gender.
        firstname: Optional first name.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/name/findName"
    querystring = {}
    if lastname:
        querystring['lastName'] = lastname
    if gender:
        querystring['gender'] = gender
    if firstname:
        querystring['firstName'] = firstname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genre(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a genre."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/music/genre"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def words(count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a list of lorem ipsum words."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/lorem/words"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def word(length: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a single word or lorem ipsum."
    length: The length of the word.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/lorem/word"
    querystring = {}
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def text(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate lorem ipsum text."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/lorem/text"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def slug(wordcount: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a lorem ipsum slug."
    wordcount: Number of words in the slug.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/lorem/slug"
    querystring = {}
    if wordcount:
        querystring['wordCount'] = wordcount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def paragraph(sentencecount: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a paragraph of lorem ipsum."
    sentencecount: The number of sentences to include in the paragraph.  Default is 3.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/lorem/paragraph"
    querystring = {}
    if sentencecount:
        querystring['sentenceCount'] = sentencecount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hex_color(basered255: int=None, basegreen255: int=None, baseblue255: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a css hex color."
    basered255: The red component. 
        basegreen255: Green component.
        baseblue255: Blue component.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/color"
    querystring = {}
    if basered255:
        querystring['baseRed255'] = basered255
    if basegreen255:
        querystring['baseGreen255'] = basegreen255
    if baseblue255:
        querystring['baseBlue255'] = baseblue255
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lines(linecount: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate lines of lorem ipsum."
    linecount: Number of lines to generate.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/lorem/lines"
    querystring = {}
    if linecount:
        querystring['lineCount'] = linecount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_name(lastname: str=None, firstname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a user name."
    lastname: Optional last name.
        firstname: Optional first name.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/userName"
    querystring = {}
    if lastname:
        querystring['lastName'] = lastname
    if firstname:
        querystring['firstName'] = firstname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_agent(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a user agent string."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/userAgent"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def protocol(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a protocol (ie http or https)."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/protocol"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def port(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a port number."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/port"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def password(len: int=None, prefix: str=None, pattern: str=None, usememorable: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a password."
    len: Length for the password.
        prefix: Prefix for the password.
        pattern: Pattern to use.
        usememorable: Create a memorable password.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/password"
    querystring = {}
    if len:
        querystring['len'] = len
    if prefix:
        querystring['prefix'] = prefix
    if pattern:
        querystring['pattern'] = pattern
    if usememorable:
        querystring['useMemorable'] = usememorable
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mac(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a mac address."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/mac"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ipv6(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an ipv6 address."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/ipv6"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an ip address."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/ip"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def http_method(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate http method (ie GET, PUT, POST, etc)"
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/httpMethod"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def example_email(lastname: str=None, firstname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an example email address."
    lastname: Optional last name.
        firstname: Optional first name.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/exampleEmail"
    querystring = {}
    if lastname:
        querystring['lastName'] = lastname
    if firstname:
        querystring['firstName'] = firstname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def email(firstname: str=None, provider: str=None, lastname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate an email address."
    firstname: Optional first name.
        provider: Optional provider (ie gmail, msn, etc)
        lastname: Optional last name.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/email"
    querystring = {}
    if firstname:
        querystring['firstName'] = firstname
    if provider:
        querystring['provider'] = provider
    if lastname:
        querystring['lastName'] = lastname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_word(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a domain word."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/domainWord"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_suffix(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a domain suffix."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/domainSuffix"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a domain name."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/internet/domainName"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transport(height: int=None, userandomize: bool=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a transport image."
    height: Height of the image. Default is 480.
        userandomize: Add a random number parameter to the returned url.
        width: Width of the image. Default is 640.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/transport"
    querystring = {}
    if height:
        querystring['height'] = height
    if userandomize:
        querystring['useRandomize'] = userandomize
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def technics(height: int=None, userandomize: bool=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a technics image."
    height: Height of the image. Default is 480.
        userandomize: Add a random number parameter to the returned url.
        width: Width of the image. Default is 640.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/technics"
    querystring = {}
    if height:
        querystring['height'] = height
    if userandomize:
        querystring['useRandomize'] = userandomize
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sports(width: int=None, height: int=None, userandomize: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a sports image."
    width: Width of the image. Default is 640.
        height: Height of the image. Default is 480.
        userandomize: Add a random number parameter to the returned url.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/sports"
    querystring = {}
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if userandomize:
        querystring['useRandomize'] = userandomize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def people(height: int=None, width: int=None, userandomize: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a people image."
    height: Height of the image. Default is 480.
        width: Width of the image. Default is 640.
        userandomize: Add a random number parameter to the returned url.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/people"
    querystring = {}
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    if userandomize:
        querystring['useRandomize'] = userandomize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nightlife(height: int=None, userandomize: bool=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a nightlife image."
    height: Height of the image. Default is 480.
        userandomize: Add a random number parameter to the returned url.
        width: Width of the image. Default is 640.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/nightlife"
    querystring = {}
    if height:
        querystring['height'] = height
    if userandomize:
        querystring['useRandomize'] = userandomize
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nature(height: int=None, width: int=None, userandomize: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a nature image."
    height: Height of the image. Default is 480.
        width: Width of the image. Default is 640.
        userandomize: Add a random number parameter to the returned url.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/image/nature"
    querystring = {}
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    if userandomize:
        querystring['useRandomize'] = userandomize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bear(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Randomly generate a bear species."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/animal/bear"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zip_code_by_state(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random zip code by state."
    state: The state to generate a zip code for.  Values:
AZ
AL
CO
NY
Etc
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/zipCodeByState"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zip_code(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random zip code."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/zipCode"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_zone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random time zone."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/timeZone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def street_suffix(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random street suffix."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/streetSuffix"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def street_prefix(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random street prefix."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/streetPrefix"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def street_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random street name."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/streetName"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def street_address(usefulladdress: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random street address."
    usefulladdress: Use a full street address.  Values:
true
false
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/streetAddress"
    querystring = {}
    if usefulladdress:
        querystring['useFullAddress'] = usefulladdress
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state_abbr(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random abbreviated state."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/stateAbbr"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random state."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/state"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def secondary_address(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random secondary address."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/secondaryAddress"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ordinal_direction(useabbr: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random ordinal direction."
    useabbr: If the result should be abbreviated.  Values:
true
false
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/ordinalDirection"
    querystring = {}
    if useabbr:
        querystring['useAbbr'] = useabbr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latitude(min: int=None, max: int=None, precision: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random latitude."
    min: Minimum value for latitude.
        max: Maximum value for latitude.
        precision: Precision for latitude.
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/latitude"
    querystring = {}
    if min:
        querystring['min'] = min
    if max:
        querystring['max'] = max
    if precision:
        querystring['precision'] = precision
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def direction(useabbr: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random direction."
    useabbr: Abbreviate the direction.  Values:
true
false
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/direction"
    querystring = {}
    if useabbr:
        querystring['useAbbr'] = useabbr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def county(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random county."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/county"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_code(alphacode: str='alpha-2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random country code."
    alphacode: Allowed values:
alpha-2
alpha-3
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/countryCode"
    querystring = {}
    if alphacode:
        querystring['alphaCode'] = alphacode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random country."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/country"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_suffix(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random city suffix."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/citySuffix"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_prefix(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random city prefix."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/cityPrefix"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random city name."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/cityName"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city(format: str=' {{address.cityPrefix}} {{name.firstName}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random city name with an optional format specifier."
    format: Examples:
{{address.cityPrefix}} {{name.firstName}}{{address.citySuffix}}
 {{address.cityPrefix}} {{name.firstName}}
{{name.firstName}}{{address.citySuffix}}
{{name.lastName}}{{address.citySuffix}}
        
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/city"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cardinal_direction(useabbr: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random cardinal direction (ie North, South, East or West). Use the useAbbr parameter to return an abbreviated result."
    
    """
    url = f"https://entreapi-faker.p.rapidapi.com/address/cardinalDirection"
    querystring = {}
    if useabbr:
        querystring['useAbbr'] = useabbr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entreapi-faker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

