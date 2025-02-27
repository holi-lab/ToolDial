import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def category(page: int=1, size: int=1000, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bulk retrieve a list of valid categories for the specified Amazon marketplace."
    region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```

        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/category/"
    querystring = {}
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_search(region: str='US', page: int=1, node_id: int=None, books_new: str=None, keywords: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for products on Amazon using search terms and keywords. Optionally use Amazon's category **node_id** to limit search to that category.
		
		Use our **GET Category API** endpoint to retrieve valid list of **node_id**(s)."
    region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```

        page: Enter the page of the results for your query.
        node_id: Enter the node id of the category.

Get a list of Amazon Category node_id from the category api endpoint.

        books_new: Enter the interval to filter for newly released books. (last-30-days, last-90-days, coming-soon)

Get a list of book search results for new releases in this time interval. Works on books only.
        keywords: Enter the keywords or search terms for the product.
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/search/"
    querystring = {}
    if region:
        querystring['region'] = region
    if page:
        querystring['page'] = page
    if node_id:
        querystring['node_id'] = node_id
    if books_new:
        querystring['books_new'] = books_new
    if keywords:
        querystring['keywords'] = keywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def best_sellers_deprecated(node_id: int, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is deprecated, use the Best Sellers by Category endpoint instead.
		
		(DEPRECATED) Request a list of Best Sellers using Amazon's category **node_id**.
		
		Use our **GET Category API** endpoint to retrieve valid list of **node_id**(s)."
    node_id: Enter the node id of the category.

Get a list of Amazon Category node_id from the category api endpoint.

        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```

        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/bestseller/"
    querystring = {'node_id': node_id, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_reviews(asin: str, sort_by: str=None, page: int=1, media_type: str=None, reviewer_type: str=None, format_type: str=None, region: str='US', filter_by_keyword: str='highly recommended', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request for a list of product reviews available for a specified Amazon product ASIN."
    sort_by: Defaults to \"helpful\" to sort by helpful reviews.

Enter \"recent\" to sort by recency.
        page: Enter the page of the results for your query.
        media_type: Enter \"all_contents\" for reviews with text, images and video.

Enter \"media_reviews_only\" for reviews with images and video only.
        reviewer_type: Defaults to \"all_reviews\" for all product reviews.

Enter \"avp_only_reviews\" for reviews with a verified purchase.
        format_type: Enter \"all_formats\" for reviews on all product variants.

Enter \"current_format\" for reviews for current product asin only.
        region: ```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        filter_by_keyword: Enter keywords to filter reviews by. e.g. \\\"highly recommend\\\"
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/product/reviews/"
    querystring = {'asin': asin, }
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if media_type:
        querystring['media_type'] = media_type
    if reviewer_type:
        querystring['reviewer_type'] = reviewer_type
    if format_type:
        querystring['format_type'] = format_type
    if region:
        querystring['region'] = region
    if filter_by_keyword:
        querystring['filter_by_keyword'] = filter_by_keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def best_sellers_by_category(node_id: int, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a list of Best Sellers using Amazon's category **node_id**.
		
		Use our **GET Category API** endpoint to retrieve valid list of **node_id**(s)."
    node_id: Enter the node id of the category.

Get a list of Amazon Category node_id from the category api endpoint.
        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/bestseller/by-category/"
    querystring = {'node_id': node_id, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movers_and_shakers_by_department(alias: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a list of Movers & Shakers using Amazon's department alias.
		
		Use our **GET Department** endpoint to retrieve valid list of **department aliases**."
    alias: Enter the department alias of the Amazon category.

Get a list of Amazon departments from the departments api endpoint.
        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/movers-and-shakers/by-department/"
    querystring = {'alias': alias, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def best_sellers_by_department(alias: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a list of Best Sellers using Amazon's department **alias**.
		
		Use our **GET Department API** endpoint to retrieve valid list of **aliases**."
    alias: Enter the department alias of the Amazon category.

Get a list of Amazon departments from the departments api endpoint.
        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/bestseller/by-department/"
    querystring = {'alias': alias, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_details(asin: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request for full product details using Amazon's ASIN."
    asin: Enter the 10-digit Amazon ASIN (e.g. B07R6V1MXW).
        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```

        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/product/"
    querystring = {'asin': asin, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_offers(asin: str, page: int=1, zip_code: str='90210', region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request for a list of product offers available for a specified Amazon product ASIN."
    asin: Enter the 10-digit Amazon ASIN (e.g. B07R6V1MXW).
        zip_code: Enter a valid zip code for the specified region. This parameter is not supported in all regions.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore

```

        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```

        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/product/offers/"
    querystring = {'asin': asin, }
    if page:
        querystring['page'] = page
    if zip_code:
        querystring['zip_code'] = zip_code
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_releases_by_department(alias: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a list of New Releases using Amazon's department **alias**.
		
		Use our **GET Department API** endpoint to retrieve valid list of **aliases**."
    alias: Enter the department alias of the Amazon category.

Get a list of Amazon departments from the departments api endpoint.
        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/new-releases/by-department/"
    querystring = {'alias': alias, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_releases_by_category(node_id: int, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a list of New Releases using Amazon's category **node_id**.
		
		Use our **GET Category API** endpoint to retrieve valid list of **node_id**(s)."
    node_id: Enter the node id of the category.

Get a list of Amazon Category node_id from the category api endpoint.
        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/new-releases/by-category/"
    querystring = {'node_id': node_id, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_wished_for_by_department(alias: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a list of Most Wished For using Amazon's department **alias**.
		
		Use our **GET Department API** endpoint to retrieve valid list of **aliases**."
    alias: Enter the department alias of the Amazon category.

Get a list of Amazon departments from the departments api endpoint.
        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/most-wished-for/by-department/"
    querystring = {'alias': alias, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_wished_for_by_category(node_id: int, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a list of Most Wished For using Amazon's category **node_id**.
		
		Use our **GET Category API** endpoint to retrieve valid list of **node_id**(s)."
    node_id: Enter the node id of the category.

Get a list of Amazon Category node_id from the category api endpoint.
        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/most-wished-for/by-category/"
    querystring = {'node_id': node_id, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_gifted_by_department(alias: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a list of Most Gifted using Amazon's department **alias**.
		
		Use our **GET Department API** endpoint to retrieve valid list of **aliases**."
    alias: Enter the department alias of the Amazon category.

Get a list of Amazon departments from the departments api endpoint.
        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/most-gifted/by-department/"
    querystring = {'alias': alias, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_gifted_by_category(node_id: int, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a list of Most Gifted using Amazon's category **node_id**.
		
		Use our **GET Category API** endpoint to retrieve valid list of **node_id**(s)."
    node_id: Enter the node id of the category.

Get a list of Amazon Category node_id from the category api endpoint.
        region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/most-gifted/by-category/"
    querystring = {'node_id': node_id, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_aliases(region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of search aliases to refine your search results for search endpoint.
		
		A search_alias parameter can be passed to the search endpoint to focus search results in that particular Amazon search index."
    region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/search/aliases"
    querystring = {'region': region, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def departments(region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of departments for use with the product ranking endpoints."
    region: Enter the 2 letter **region code** of the Amazon marketplace.

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
```
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/departments/"
    querystring = {}
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_summaries(asins: str, region: str='US', page: int=1, all: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of product summaries from Amazon using ASINs. 
		
		This is a lightweight endpoint for requesting a list of  basic product data such as buy box price, sales, availability and rating in bulk. This is great for updating and tracking many products and is designed to be user friendly for spreadsheets such as Google Sheets.
		
		Notes:
		- The maximum limit for number of ASINs per request is 300 ASINs. 
		- The maxmum number of results returned is 48 results per request, you need to paginate the request if there is more than 48 ASINs in the request.
		- If a product ASIN doesn't exist anymore, it will not be returned in the results.
		- If the ASIN has changed for a particular product, the new ASIN will be returned."
    asins: Enter ASINs separated by a comma. (e.g. B07R6V1MXW,B07QHC1FS6,1501110365)
        region: Enter the 2 letter **region code** of the Amazon marketplace. Default value is \\\"US\\\".

**Supported Regions**:
```
US - www.amazon.com    - United States  
CA - www.amazon.ca     - Canada
MX - www.amazon.com.mx - Mexico
BR - www.amazon.com.br - Brazil
AU - www.amazon.com.au - Australia
FR - www.amazon.fr     - France
DE - www.amazon.de     - Germany
IT - www.amazon.it     - Italy
NL - www.amazon.nl     - Netherlands
ES - www.amazon.es     - Spain
SE - www.amazon.se     - Sweden
GB - www.amazon.co.uk  - United Kingdom
PL - www.amazon.pl     - Poland
AE - www.amazon.ae     - United Arab Emirates
SA - www.amazon.sa     - Saudi Arabia
TR - www.amazon.tr     - Turkey
IN - www.amazon.in     - India
JP - www.amazon.co.jp  - Japan
SG - www.amazon.sg     - Singapore
CN - www.amazon.cn     - China
        page: Enter the page of the results for your query.
        all: Return all results at once. This parameter causes the endpoint to paginate all available pages and sort products by request order.

Set this parameter `all=1` if you want to grab all results in bulk. Default is set to 0.
        
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/product/summaries/"
    querystring = {'asins': asins, }
    if region:
        querystring['region'] = region
    if page:
        querystring['page'] = page
    if all:
        querystring['all'] = all
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def related_product_asins(asin: str, region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request for ASINs related to a Amazon product."
    
    """
    url = f"https://parazun-amazon-data.p.rapidapi.com/product/related-asins/"
    querystring = {'asin': asin, 'region': region, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parazun-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

