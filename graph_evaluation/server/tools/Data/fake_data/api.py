import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_fake_images_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Images endpoint Contains the following attributes.
		1. image.
		2. avatar.
		3. imageUrl .
		4. abstract.
		5. animals.
		6. business.
		7. cats.
		8. city.
		9. food.
		10. nightlife.
		11. fashion.
		12. people.
		13. nature.
		14. sports.
		15. technics.
		16. transport.
		17. dataUri."
    
    """
    url = f"https://fake-data.p.rapidapi.com/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_address_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Address endpoint Contains the following attributes.
		1. city.
		2. cityPrefix.
		3. citySuffix.
		4. streetName.
		5. streetAddress.
		6. streetSuffix.
		7. streetPrefix.
		8. secondaryAddress.
		9. county.
		10. country.
		11. countryCode.
		12. state.
		13. stateAbbr.
		14. latitude.
		15. longitude.
		16. building_number."
    
    """
    url = f"https://fake-data.p.rapidapi.com/address"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_internet_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Internet endpoint Contains the following attributes.
		1. avatar.
		2. email.
		3. exampleEmail.
		4. userName.
		5. protocol.
		6. url.
		7. domainName.
		8. domainSuffix.
		9. domainWord.
		10. ip.
		11. ipv6.
		12. userAgent.
		13. mac.
		14. password."
    
    """
    url = f"https://fake-data.p.rapidapi.com/internet"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_miscellaneous_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mics endpoint Contains quite a lot of miscellaneous attributes. which might come in handy.
		1. country_code.
		2. language_code.
		3. locale.
		4. currency.
		5. currency_code.
		6. currency_symbol
		7. currency_name.
		8. mime_type.
		9. boolean.
		10. uuid.
		11. number.
		12. alphaNumeric.
		13. objectElement.
		14. words.
		15. image.
		16. arrayElement.
		17. abbreviation.
		18. adjective.
		19. noun.
		20. verb.
		21. ingverb.
		22. phrase.
		23. color_name.
		24. safe_color_name.
		25. rgb_hex.
		26. rgb_array."
    
    """
    url = f"https://fake-data.p.rapidapi.com/misc"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_person_personal_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Person endpoint Contains the following attributes.
		1. name.
		2. title.
		3. username.
		4. first_name.
		5. last_name.
		6. full_name.
		7. phoneFormats.
		8. phoneNumber.
		9. phoneNumberFormat.
		10. password.
		11. name_prefix.
		12. name_suffix.
		13. company_name.
		14. company_suffix.
		15. catch_phrase.
		16. jobArea. _ jobtype"
    
    """
    url = f"https://fake-data.p.rapidapi.com/person"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_payment_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "payments endpoint Contains the following attributes.
		1. account.
		2. accountName.
		3. mask.
		4. amount.
		5. transactionType.
		6. currencyCode.
		7. currencyName.
		8. currencySymbol.
		9. bitcionAddress.
		10. bic.
		11. iban.
		12. card_type.
		13. card_number.
		14. card_exp.
		15. card_data."
    
    """
    url = f"https://fake-data.p.rapidapi.com/payments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_number_math_related_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Number endpoint Contains the following attributes.
		1. random.
		2. integer.
		3. double.
		4. array_of_digits.
		5. array_of_integers.
		6. array_of_doubles.
		7. coin_flip."
    
    """
    url = f"https://fake-data.p.rapidapi.com/number"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_system_computer_related_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "System endpoint Contains the following attributes.
		1. fileExt.
		2. commonFileExt.
		3. commonFileName.
		4. commonFileType.
		5. fileName.
		6. filePath.
		7. filetype.
		8. mimeType.
		9. semver.
		10. directoryPath."
    
    """
    url = f"https://fake-data.p.rapidapi.com/system"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_lorem_ipsum_paragraphs_of_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lorem endpoint Contains the following attributes.
		1. word.
		2. words.
		3. sentence.
		4. slug.
		5. sentences.
		6. paragraph.
		7. paragraphs.
		8. text.
		9. lines.
		10. title."
    
    """
    url = f"https://fake-data.p.rapidapi.com/lorem"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_helpers_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Helpers endpoint Contains the following attributes.
		1. randomize.
		2. shuffle.
		3. replaceSymbolWithNumber.
		4. replaceSymbols.
		5. mustache.
		6. createCard.
		7. contextualCard.
		8. userCard.
		9. createTransaction."
    
    """
    url = f"https://fake-data.p.rapidapi.com/helpers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_date_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Date endpoint Contains the following attributes.
		1. unix_time .
		2. moment.
		3. date.
		4. time.
		5. century.
		6. am_pm.
		7. day_of_month.
		8. day_of_year.
		9. month_number.
		10. month_name.
		11. year.
		12. timezone."
    
    """
    url = f"https://fake-data.p.rapidapi.com/date"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fake_commerce_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Commerce endpoint Contains the following attributes.
		1. department.
		2. productName.
		3. price.
		4. productAdjective.
		5. productMaterial.
		6. product.
		7. suffixes.
		8. companyName.
		9. companySuffix.
		10. catchPhrase.
		11. bs.
		12. catchPhraseAdjective.
		13. catchPhraseDescriptor.
		14. catchPhraseNoun.
		15. bsAdjective.
		16. bsBuzz.
		17. bsNoun.
		18. column.
		19. type.
		20. collation.
		21. engine."
    
    """
    url = f"https://fake-data.p.rapidapi.com/commerce"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

