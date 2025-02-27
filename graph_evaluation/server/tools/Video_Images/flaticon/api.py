import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def style(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you go get detailed information about a style, the design format that an icon or a pack has. Styles are the different versions in which an icon a given family may appear. For example, an icon may be available in lineal, color, or lineal color styles."
    
    """
    url = f"https://flaticon.p.rapidapi.com/style/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get detailed information about a category. Categories are the different themes in which icons are grouped at Flaticon."
    
    """
    url = f"https://flaticon.p.rapidapi.com/category/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totalpacks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a number stating the full quantity of packs available at Flaticon. The number refreshes periodically according to the content available at flaticon.com. Packs are sorted into two groups: PREMIUM packs and SELECTION packs. PREMIUM packs are those that contain one or more PREMIUM icons."
    
    """
    url = f"https://flaticon.p.rapidapi.com/total/packs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(page: int=None, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a paged list with information about Flaticon categories. Categories are the different themes in which icons are grouped at Flaticon."
    page: Page about which the request is made. In case it doesn’t exist, it will return page 1.
        limit: Number of max. authors returned per request. In case it isn’t requested, it will return 100 categories by default.
        
    """
    url = f"https://flaticon.p.rapidapi.com/categories"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totalpackspremium(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a number stating the full quantity of PREMIUM packs available at Flaticon. The number refreshes periodically according to the content available at flaticon.com. PREMIUM packs are those that contain one or more PREMIUM icons. "
    
    """
    url = f"https://flaticon.p.rapidapi.com/total/packs/premium"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download(format: str, is_id: int, color: str=None, size: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to download the selected icon with the selected format."
    format: Item format to download. Is possible to choose between 'svg' and 'png' formats. If no format is sent, it will return the item in 'png' format.

        color: Monocolor Icons can be downloaded in different colors. Color property has to be a valid hexadecimal color. If no color is sent, it will return the icon in black.

        size: Size to download the selected Icon. the valid sizes are: 16, 24, 32, 64, 128, 256, 512. If no size is sent it will return the icon with a size of 32px

        
    """
    url = f"https://flaticon.p.rapidapi.com/item/icon/download/{is_id}"
    querystring = {'format': format, }
    if color:
        querystring['color'] = color
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags(page: int=None, limit: int=100, havingstickers: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a paged list with information about Flaticon tags that icons or packs have."
    page: Page about which the request is made. In case it doesn’t exist, it will return page 1.
        limit: Number of max. authors returned per request. In case it isn’t requested, it will return 100 styles by default.
        havingstickers: Get tags that have stickers
        
    """
    url = f"https://flaticon.p.rapidapi.com/tags"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if havingstickers:
        querystring['havingStickers'] = havingstickers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tag(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you go get detailed information about a tag."
    
    """
    url = f"https://flaticon.p.rapidapi.com/tag/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totalpacksselection(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a number stating the full quantity of SELECTION packs available at Flaticon. The number refreshes periodically according to the content available at flaticon.com. SELECTION packs are those that contain only SELECTION icons."
    
    """
    url = f"https://flaticon.p.rapidapi.com/total/packs/selection"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totaliconsselection(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a number stating the full quantity of SELECTION icons available at Flaticon. The number refreshes periodically according to the content available at flaticon.com."
    
    """
    url = f"https://flaticon.p.rapidapi.com/total/icons/selection"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchpacks(q: str, orderby: str, stylename: str=None, page: int=None, categoryname: str=None, icontype: str=None, limit: int=100, familyid: int=None, color: int=None, tagsid: str=None, categoryid: int=None, url: str=None, stroke: int=None, familyname: str=None, designerid: int=None, styleid: int=None, designername: str=None, packid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a paged list with the results of the search. Packs returned by the search may be SELECTION packs or PREMIUM packs, according to the content."
    q: term to search for
        orderby: Order by in results from Flaticon API. You can choose between order the results between 'priority' or 'added'. Priority order will return items ordered by an algorithm based on the downloads ratio. Added order will return items ordered by the date they were added to Flaticon.
For example: `https://api.flaticon.com/v2/search/packs/added?q=media&stroke=2&styleName=gradient&packId=1554346&limit=15`

        stylename: Filter the results by the name of style.
        page: Page number
        categoryname: Filter the results by category name.
        icontype: Filter the results by icon type, valid values are `standard` and  `stickers`
        limit: Number of max. items returned per request. In case it isn’t requested, it will return 100 packs by default.
        familyid: Filter the results by family id.
        color: Filter the results by color id. Color id `1` means monocolor. Color id `2` means multicolor.
        tagsid: Filter the results by a list of tag ids separated by comma
        categoryid: Filter the results by category id.
        url: pack url to search for
        stroke: Filter the results by stroke id. Stroke id `1` means filled. Stroke id `2` means linear.
        familyname: Filter the results by family name.
        designerid: Filter the results by designer id.
        styleid: Filter the results by style id.
        designername: Filter the results by designer name.
        packid: Filter the results by pack id.
        
    """
    url = f"https://flaticon.p.rapidapi.com/search/packs/{orderby}"
    querystring = {'q': q, }
    if stylename:
        querystring['styleName'] = stylename
    if page:
        querystring['page'] = page
    if categoryname:
        querystring['categoryName'] = categoryname
    if icontype:
        querystring['iconType'] = icontype
    if limit:
        querystring['limit'] = limit
    if familyid:
        querystring['familyId'] = familyid
    if color:
        querystring['color'] = color
    if tagsid:
        querystring['tagsId'] = tagsid
    if categoryid:
        querystring['categoryId'] = categoryid
    if url:
        querystring['url'] = url
    if stroke:
        querystring['stroke'] = stroke
    if familyname:
        querystring['familyName'] = familyname
    if designerid:
        querystring['designerId'] = designerid
    if styleid:
        querystring['styleId'] = styleid
    if designername:
        querystring['designerName'] = designername
    if packid:
        querystring['packId'] = packid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchicons(q: str, orderby: str, packid: int=None, categoryname: str=None, designerid: int=None, page: int=None, categoryid: int=None, tagsid: str=None, familyid: int=None, url: str=None, designername: str=None, color: int=None, icontype: str=None, stroke: int=None, familyname: str=None, limit: int=100, styleid: int=None, stylename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a paged list with the results of the search. Icons returned by the search may be SELECTION icons or PREMIUM icons. By default, it will return a list order by priority. If no order is set, it will return items order by priority. In order to avoid 403 errors when svg icon is used to show the icon in result lists you must not store data in cache more than 15 minutes"
    q: term to search for
        orderby: Order by in results from Flaticon API. You can choose between order the results between 'priority' or 'added'. Priority order will return items ordered by an algorithm based on the downloads ratio. Added order will return items ordered by the date they were added to Flaticon.
For example: `https://api.flaticon.com/v2/search/icons/added?q=arrow&stroke=2&styleName=gradient&packId=1554346&limit=15`

        packid: Filter the results by pack id.
        categoryname: Filter the results by category name.
        designerid: Filter the results by designer id.
        page: Page number
        categoryid: Filter the results by category id.
        tagsid: Filter the results by a list of tag ids separated by comma
        familyid: Filter the results by family id.
        url: icon url to search for
        designername: Filter the results by designer name.
        color: Filter the results by color id. Color id `1` means monocolor. Color id `2` means multicolor.
        icontype: Filter the results by icon type, valid values are `standard` and `stickers`
        stroke: Filter the results by stroke id. Stroke id `1` means filled. Stroke id `2` means linear.
        familyname: Filter the results by family name.
        limit: Number of max. items returned per request. In case it isn’t requested, it will return 100 packs by default.
        styleid: Filter the results by style id.
        stylename: Filter the results by the name of style.
        
    """
    url = f"https://flaticon.p.rapidapi.com/search/icons/{orderby}"
    querystring = {'q': q, }
    if packid:
        querystring['packId'] = packid
    if categoryname:
        querystring['categoryName'] = categoryname
    if designerid:
        querystring['designerId'] = designerid
    if page:
        querystring['page'] = page
    if categoryid:
        querystring['categoryId'] = categoryid
    if tagsid:
        querystring['tagsId'] = tagsid
    if familyid:
        querystring['familyId'] = familyid
    if url:
        querystring['url'] = url
    if designername:
        querystring['designerName'] = designername
    if color:
        querystring['color'] = color
    if icontype:
        querystring['iconType'] = icontype
    if stroke:
        querystring['stroke'] = stroke
    if familyname:
        querystring['familyName'] = familyname
    if limit:
        querystring['limit'] = limit
    if styleid:
        querystring['styleId'] = styleid
    if stylename:
        querystring['styleName'] = stylename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pack(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get detailed information about a pack. A pack is a set of icons that share a theme. Besides, any given pack may be found in one or many styles associated or not with a family."
    
    """
    url = f"https://flaticon.p.rapidapi.com/item/pack/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def packs(q: str, orderby: str, designername: str=None, page: int=None, stylename: str=None, categoryname: str=None, categoryid: int=None, url: str=None, stroke: int=None, styleid: int=None, packid: int=None, tagsid: str=None, familyname: str=None, familyid: int=None, color: int=None, icontype: str=None, designerid: int=None, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a paged list with information about the packs available at Flaticon. A pack is a set of icons that share a theme. Besides, any given pack may be found in one or many styles associated or not with a family.  By default, it will return a list order by priority. If no order is set, it will return items order by priority."
    q: term to search for
        orderby: Order by in results from Flaticon API. You can choose between order the results between 'priority' or 'added'. Priority order will return items ordered by an algorithm based on the downloads ratio. Added order will return items ordered by the date they were added to Flaticon.
For example: `https://api.flaticon.com/v2/search/icons/added?q=arrow&stroke=2&styleName=gradient&packId=1554346&limit=15`

        designername: Filter the results by designer name.
        page: Page number
        stylename: Filter the results by the name of style.
        categoryname: Filter the results by category name.
        categoryid: Filter the results by category id.
        url: icon url to search for
        stroke: Filter the results by stroke id. Stroke id `1` means filled. Stroke id `2` means linear.
        styleid: Filter the results by style id.
        packid: Filter the results by pack id.
        tagsid: Filter the results by a list of tag ids separated by comma
        familyname: Filter the results by family name.
        familyid: Filter the results by family id.
        color: Filter the results by color id. Color id `1` means monocolor. Color id `2` means multicolor.
        icontype: Filter the results by icon type, valid values are `standard` and `stickers`
        designerid: Filter the results by designer id.
        limit: Number of max. items returned per request. In case it isn’t requested, it will return 100 packs by default.
        
    """
    url = f"https://flaticon.p.rapidapi.com/items/packs/{orderby}"
    querystring = {'q': q, }
    if designername:
        querystring['designerName'] = designername
    if page:
        querystring['page'] = page
    if stylename:
        querystring['styleName'] = stylename
    if categoryname:
        querystring['categoryName'] = categoryname
    if categoryid:
        querystring['categoryId'] = categoryid
    if url:
        querystring['url'] = url
    if stroke:
        querystring['stroke'] = stroke
    if styleid:
        querystring['styleId'] = styleid
    if packid:
        querystring['packId'] = packid
    if tagsid:
        querystring['tagsId'] = tagsid
    if familyname:
        querystring['familyName'] = familyname
    if familyid:
        querystring['familyId'] = familyid
    if color:
        querystring['color'] = color
    if icontype:
        querystring['iconType'] = icontype
    if designerid:
        querystring['designerId'] = designerid
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icons(orderby: str, q: str, designerid: int=None, page: int=None, categoryname: str=None, styleid: int=None, limit: int=100, stylename: str=None, designername: str=None, stroke: int=None, familyname: str=None, categoryid: int=None, tagsid: str=None, packid: int=None, familyid: int=None, icontype: str=None, color: int=None, url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a paged list with information about the icons available at Flaticon. By default, it will return a list order by priority. If no order is set, it will return items order by priority. In order to avoid 403 errors when svg icon is used to show the icon in result lists you must not store data in cache more than 15 minutes"
    orderby: Order by in results from Flaticon API. You can choose between order the results between 'priority' or 'added'. Priority order will return items ordered by an algorithm based on the downloads ratio. Added order will return items ordered by the date they were added to Flaticon.
For example: `https://api.flaticon.com/v2/search/icons/added?q=arrow&stroke=2&styleName=gradient&packId=1554346&limit=15`

        q: term to search for
        designerid: Filter the results by designer id.
        page: Page number
        categoryname: Filter the results by category name.
        styleid: Filter the results by style id.
        limit: Number of max. items returned per request. In case it isn’t requested, it will return 100 packs by default.
        stylename: Filter the results by the name of style.
        designername: Filter the results by designer name.
        stroke: Filter the results by stroke id. Stroke id `1` means filled. Stroke id `2` means linear.
        familyname: Filter the results by family name.
        categoryid: Filter the results by category id.
        tagsid: Filter the results by a list of tag ids separated by comma
        packid: Filter the results by pack id.
        familyid: Filter the results by family id.
        icontype: Filter the results by icon type, valid values are `standard` and `stickers`
        color: Filter the results by color id. Color id `1` means monocolor. Color id `2` means multicolor.
        url: icon url to search for
        
    """
    url = f"https://flaticon.p.rapidapi.com/items/icons/{orderby}"
    querystring = {'q': q, }
    if designerid:
        querystring['designerId'] = designerid
    if page:
        querystring['page'] = page
    if categoryname:
        querystring['categoryName'] = categoryname
    if styleid:
        querystring['styleId'] = styleid
    if limit:
        querystring['limit'] = limit
    if stylename:
        querystring['styleName'] = stylename
    if designername:
        querystring['designerName'] = designername
    if stroke:
        querystring['stroke'] = stroke
    if familyname:
        querystring['familyName'] = familyname
    if categoryid:
        querystring['categoryId'] = categoryid
    if tagsid:
        querystring['tagsId'] = tagsid
    if packid:
        querystring['packId'] = packid
    if familyid:
        querystring['familyId'] = familyid
    if icontype:
        querystring['iconType'] = icontype
    if color:
        querystring['color'] = color
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def styles(limit: int=100, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a paged list with information about Flaticon styles, design formats that icons or packs have. Styles are the different versions in which an icon from a given family may appear. For example, an icon may be available in lineal, color, or lineal color styles."
    limit: Number of max. authors returned per request. In case it isn’t requested, it will return 100 styles by default.
        page: Page about which the request is made. In case it doesn’t exist, it will return page 1.
        
    """
    url = f"https://flaticon.p.rapidapi.com/styles"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totalicons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a number stating the full quantity of icons available at Flaticon. The number refreshes periodically according to the content available at flaticon.com."
    
    """
    url = f"https://flaticon.p.rapidapi.com/total/icons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totaliconspremium(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get a number stating the full quantity of PREMIUM icons available at Flaticon. The number refreshes periodically according to the content available at flaticon.com."
    
    """
    url = f"https://flaticon.p.rapidapi.com/total/icons/premium"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def icon(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method that allows you to get detailed information about an icon."
    
    """
    url = f"https://flaticon.p.rapidapi.com/item/icon/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flaticon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

