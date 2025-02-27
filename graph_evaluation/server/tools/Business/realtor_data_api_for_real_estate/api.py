import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def realtorpropertylist(state_code: str, offset: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will give you the Realtor Property List with the data as given below, if you provide the city, state_code and offset.
		Parameters:
		city: San Jose or New York
		state_code: CA or NY
		offset: min = 0 and max can be depends on the total number of results. but the maximum number of results you can see at a time is 42. So you need to keep increasing the offset value by 42 to achieve a pagination.  To achieve pagination you have to start the offset from 0 and increase the offset by 42.
		
		`{
		    "agent": [
		      {
		        "office_name": "CENTURY21 REAL ESTATE ALLIANCE"
		      },
		      {
		        "office_name": null
		      }
		    ],
		    "branding": [
		      {
		        "name": "CENTURY 21 Real Estate Alliance",
		        "photo": null,
		        "type": "Office"
		      }
		    ],
		    "last_update_date": "2022-12-06T01:24:16Z",
		    "list_date": "2022-08-23T23:24:23Z",
		    "list_price": 875000,
		    "listing_id": "2946989531",
		    "location": {
		      "address": {
		        "city": "San Francisco",
		        "coordinate": {
		          "lat": 37.71979,
		          "lon": -122.462898
		        },
		        "line": "9 Garfield St",
		        "postal_code": "94132",
		        "state": "California",
		        "state_code": "CA"
		      },
		      "county": {
		        "fips_code": "06075",
		        "name": "San Francisco"
		      },
		      "street_view_url": "https://maps.googleapis.com/maps/api/streetview?channel=rdc-streetview&client=gme-movesalesinc&location=9%20Garfield%20St%2C%20San%20Francisco%2C%20CA%2094132&size=640x480&source=outdoor&signature=NniMw06UKhWMjlwyIN-dwOajrxo="
		    },
		    "open_house_description": "No Open House Available",
		    "other_listings": {
		      "rdc": [
		        {
		          "listing_id": "2946989531",
		          "listing_key": null,
		          "primary": true,
		          "status": "for_sale"
		        }
		      ]
		    },
		    "permalink": "https://www.realtor.com/realestateandhomes-detail/9-Garfield-St_San-Francisco_CA_94132_M23343-72866",
		    "photos": [
		      {
		        "href": "https://ap.rdcpix.com/690f73cb78c5a22ccb272def0e3435fel-b1134851845s.jpg"
		      },
		      {
		        "href": "https://ap.rdcpix.com/690f73cb78c5a22ccb272def0e3435fel-b145756411s.jpg"
		      }
		    ],
		    "price_reduced_amount": null,
		    "primary_photo": "https://ap.rdcpix.com/690f73cb78c5a22ccb272def0e3435fel-b1134851845s.jpg",
		    "property_description": {
		      "baths": 1,
		      "baths_1qtr": null,
		      "baths_3qtr": null,
		      "baths_full": 1,
		      "baths_half": null,
		      "beds": 0,
		      "garage": 2,
		      "lot_sqft": 2539,
		      "name": null,
		      "sold_date": "2017-08-17",
		      "sold_price": 45373,
		      "sqft": 1094,
		      "stories": null,
		      "sub_type": null,
		      "type": "single_family",
		      "year_built": 1947
		    },
		    "property_id": "2334372866",
		    "source": "BDCA",
		    "status": "for_sale",
		    "tax_record": "6A966F0C987E7C5A1DB0D29A6B22116A",
		    "type": "mls",
		    "virtual_tour_link": "No Virtual Tour Available"
		  }`"
    
    """
    url = f"https://realtor-data-api-for-real-estate.p.rapidapi.com/realtor_data/property/"
    querystring = {'state_code': state_code, 'offset': offset, 'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-data-api-for-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def realtorschoollist(school_level: str, page: str, city: str, state_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will give you the Realtor School List with the data as given below, if you provide the city, state_code and page.
		Parameters:
		city: San Jose or New York etc
		state_code: CA or NY etc
		school_level: It can be either one of them (elementary, middle, high, private & charter)
		page: min = 1 and max can be depends on the total number of results. 
		
		`{
		    "coordinates": {
		      "lat": 37.247605,
		      "lon": -121.946069
		    },
		    "funding_type": "public",
		    "grades": [
		      "K",
		      "1",
		      "2",
		      "3",
		      "4",
		      "5"
		    ],
		    "id": "078653021",
		    "location": {
		      "city": "San Jose",
		      "city_slug_id": "San-Jose_CA",
		      "state": "CA"
		    },
		    "name": "Carlton Elementary School",
		    "parent_rating": 4,
		    "rating": 9,
		    "review_count": 27,
		    "school url": "https://www.realtor.com/local/schools/Carlton-Elementary-School-078653021"
		  }`"
    
    """
    url = f"https://realtor-data-api-for-real-estate.p.rapidapi.com/realtor_data/schools/"
    querystring = {'school_level': school_level, 'page': page, 'city': city, 'state_code': state_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-data-api-for-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def realtoragentlist(page: str, state_code: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will give you the Realtor Agent List with the data as given below, if you provide the city, state_code and page.
		Parameters:
		city: San Jose or New York
		state_code: CA or NY
		page: min = 1 and max can be depends on the total number of results. 
		
		`{
		    "agent_address": {
		      "city": "PLEASANTON",
		      "country": "US",
		      "line": "",
		      "line2": "",
		      "postal_code": "94588",
		      "state": "",
		      "state_code": "CA"
		    },
		    "agent_bio": "Mark Lafferty Office: 925-216-6203 \nEmail Me Keller Williams Tri-Valley Realty\n2300 First St. #316\nLivermore, CA 94550\n\nExperience\nAs an agent who's an expert in this local area, I bring a wealth of knowledge and expertise about buying and selling real estate here. It's not the same everywhere, so you need someone you can trust for up-to-date information. I am eager to serve you. Here are some of the things I can do for you:\n\nFind Your Next Home\nYou need someone who knows this area inside and out! I can work with you to find the right home at the right price for you, including all the neighborhood amenities that matter - not to mention the essential criteria you have for your ideal home\n\nSell a Home\nWhen it's time to move, you need someone who will advertise your home, show to prospective buyers, negotiate the purchase contract, arrange financing, oversee the inspections, handle all necessary paperwork and supervise the closing. I can take care of everything you need, from start to close.\n\nConsult on Home Selling Tactics\nOftentimes buyers don't visualize living in your home the way you do. I can make your home attractive to its ideal audience - which can help you get top dollar. Things like staging the home, making repairs or minor improvements, or even simply painting the walls can be the difference between a home resting on the market and one that's sold fast.\n..",
		    "agent_languages": [],
		    "agent_mls": [
		      {
		        "abbreviation": "FAR_19911A4E",
		        "license_number": "01344268",
		        "member": {
		          "id": "FAR_1399152E133D152E0030"
		        },
		        "primary": false,
		        "type": "A"
		      },
		      {
		        "abbreviation": "FAR_19ED1A4E",
		        "license_number": "01344268",
		        "member": {
		          "id": "FAR_1399152E133D152E0030"
		        },
		        "primary": false,
		        "type": "A"
		      },
		      {
		        "abbreviation": "FAR_1A021A4E",
		        "license_number": "01344268",
		        "member": {
		          "id": "FAR_1399152E133D152E0030"
		        },
		        "primary": true,
		        "type": "A"
		      },
		      {
		        "abbreviation": "FAR_20971A4E",
		        "license_number": "01344268",
		        "member": {
		          "id": "FAR_12D2140114631531"
		        },
		        "primary": false,
		        "type": "A"
		      }
		    ],
		    "agent_photo": "http://p.rdcpix.com/v01/aaa120200-c0o.jpg",
		    "agent_rating": 0,
		    "agent_recommendations_count": 0,
		    "agent_reviews_count": 0,
		    "agent_served_areas": [
		      {
		        "name": "Fremont",
		        "state_code": "CA"
		      },
		      {
		        "name": "Livermore",
		        "state_code": "CA"
		      },
		      {
		        "name": "Pleasanton",
		        "state_code": "CA"
		      },
		      {
		        "name": "Sunol",
		        "state_code": "CA"
		      },
		      {
		        "name": "Milpitas",
		        "state_code": "CA"
		      },
		      {
		        "name": "San Jose",
		        "state_code": "CA"
		      },
		      {
		        "name": "Mount Hamilton",
		        "state_code": "CA"
		      }
		    ],
		    "agent_slogan": "",
		    "agent_specializations": [
		      {
		        "name": "Buyer"
		      },
		      {
		        "name": "Commercial"
		      },
		      {
		        "name": "First Time"
		      },
		      {
		        "name": "Golf"
		      },
		      {
		        "name": "Investment Properties"
		      },
		      {
		        "name": "Land"
		      },
		      {
		        "name": "Ranch"
		      },
		      {
		        "name": "Seller"
		      },
		      {
		        "name": "Vinyard"
		      },
		      {
		        "name": "Winery"
		      }
		    ],
		    "agent_team_details": {
		      "is_team_member": false
		    },
		    "agent_type": [
		      "buyer",
		      "seller"
		    ],
		    "agent_website": "http://www.marklafferty.com",
		    "for_sale": {
		      "count": 4,
		      "last_listing_date": "2023-01-31T01:52:43Z",
		      "max": 1970000,
		      "min": 849950
		    },
		    "is_realtor": true,
		    "joined": [
		      0,
		      0
		    ],
		    "license": [
		      {
		        "country": "US",
		        "license_number": "01344268",
		        "state_code": "CA"
		      }
		    ],
		    "marketing_areas": [
		      {
		        "city_state": "Fremont_CA",
		        "name": "Fremont",
		        "state_code": "CA"
		      },
		      {
		        "city_state": "Sunol_CA",
		        "name": "Sunol",
		        "state_code": "CA"
		      },
		      {
		        "city_state": "Livermore_CA",
		        "name": "Livermore",
		        "state_code": "CA"
		      },
		      {
		        "city_state": "Pleasanton_CA",
		        "name": "Pleasanton",
		        "state_code": "CA"
		      },
		      {
		        "city_state": "Milpitas_CA",
		        "name": "Milpitas",
		        "state_code": "CA"
		      },
		      {
		        "city_state": "San Jose_CA",
		        "name": "San Jose",
		        "state_code": "CA"
		      },
		      {
		        "city_state": "East Foothills_CA",
		        "name": "East Foothills",
		        "state_code": "CA"
		      },
		      {
		        "city_state": "Alum Rock_CA",
		        "name": "Alum Rock",
		        "state_code": "CA"
		      },
		      {
		        "city_state": "Mount Hamilton_CA",
		        "name": "Mount Hamilton",
		        "state_code": "CA"
		      }
		    ],
		    "office_address": {
		      "city": "PLEASANTON",
		      "country": "US",
		      "line": "",
		      "line2": "",
		      "postal_code": "94588",
		      "state": "",
		      "state_code": "CA"
		    },
		    "office_fulfillment_id": 1424399,
		    "office_mls": [
		      {
		        "abbreviation": "FAR_19911A4E",
		        "license_number": "",
		        "member": {
		          "id": "FAR_2098221F12D3"
		        },
		        "primary": false,
		        "type": "O"
		      },
		      {
		        "abbreviation": "FAR_19ED1A4E",
		        "license_number": "",
		        "member": {
		          "id": "FAR_2098221F12D3"
		        },
		        "primary": false,
		        "type": "O"
		      },
		      {
		        "abbreviation": "FAR_1A021A4E",
		        "license_number": "",
		        "member": {
		          "id": "FAR_2098221F12D3"
		        },
		        "primary": true,
		        "type": "O"
		      },
		      {
		        "abbreviation": "FAR_20971A4E",
		        "license_number": "",
		        "member": {
		          "id": "FAR_153013A10033"
		        },
		        "primary": false,
		        "type": "O"
		      }
		    ],
		    "office_name": "KELLER WILLIAMS TRI-VALLEY",
		    "office_party_id": "3145631",
		    "office_phone": [
		      {
		        "ext": "",
		        "number": "(925) 397-4200",
		        "type": "Office"
		      }
		    ],
		    "office_website": "WWW.PIERCETHEMARKET.COM",
		    "party_id": 4232697,
		    "person_name": "MARK M. LAFFERTY",
		    "phones": [
		      {
		        "ext": "",
		        "number": "(925) 216-6203",
		        "type": "Mobile"
		      }
		    ],
		    "position": "Agent",
		    "profile_url": "https://www.realtor.com/realestateagents/MARK-M.-LAFFERTY_PLEASANTON_CA_135850_943184397",
		    "profile_url_id": "https://www.realtor.com/realestateagents/56b01aefbb954c01006a3382",
		    "recently_sold": {
		      "count": 30,
		      "last_sold_date": "2023-01-13",
		      "max": 2850000,
		      "min": 630000
		    },
		    "served_zip_codes": [
		      "94539",
		      "94550",
		      "94551",
		      "94566",
		      "94586",
		      "95035",
		      "95036",
		      "95101",
		      "95103",
		      "95106",
		      "95108",
		      "95109",
		      "95110",
		      "95111",
		      "95112",
		      "95113",
		      "95115",
		      "95116",
		      "95121",
		      "95122",
		      "95127",
		      "95131",
		      "95132",
		      "95133",
		      "95134",
		      "95135",
		      "95138",
		      "95140",
		      "95148",
		      "95150",
		      "95151",
		      "95152",
		      "95153",
		      "95154",
		      "95155",
		      "95156",
		      "95157",
		      "95158",
		      "95159"
		    ],
		    "types": "agent",
		    "video_url": null
		  }`"
    
    """
    url = f"https://realtor-data-api-for-real-estate.p.rapidapi.com/realtor_data/agents/"
    querystring = {'page': page, 'state_code': state_code, 'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-data-api-for-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

