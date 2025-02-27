import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def file_upload_get_presigned_url_getgeneratesecureurlforupload(encrypt: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Description:** This method generates links to upload your local file to. Use this `presignedUrl` from the response to upload your file. Once you upload your file to this `presignedUrl` using `PUT`, you can use `url` link to access the uploaded file.
		
		With this method you can upload files up to 2GB in size. Please note that to process these files you should use `async=true` mode with data extraction and tools endpoints along with `/job/check` to check status of background jobs you create.
		
		
		**Steps to Upload File:**
		
		
		- First, call `/file/upload/get-presigned-url`. It will generate **link for uploading** (`presignedUrl`) and **final link** (`url`)
		- Now send your file to the `presignedUrl` link using `PUT` method within next 30 minutes. 
		- Oncce finished, use `url` to access the file you have just uploaded.
		
		
		**Important**: all uploaded files are considered to be temporary files and are automatically permanently removed after 1 hour.
		
		**Status Errors**
		
		| Code	| Description|
		|-- |--
		|`200`| The request has succeeded|
		|`400`| bad input parameters|
		|`401`|	unauthorized|
		|`403`|	not enough credits|
		|`405`|	Timeout error. To process large documents or files please use asynchronous mode ( set `async` parameter to true) and then check the status using `/job/check` endpoint. If file contains many pages then specify a page range using pages parameter. The number of pages of the document can be obtained using the endpoint `/pdf/info`|
		
		**Example**
		
		Sample Request:
		
		! Don't forget to set `x-api-key` url param or http header param (preferred) to API key, get yours [here](https://app.pdf.co/signup)
		
		```
		GET
		https://api.pdf.co/v1/file/upload/get-presigned-url?name=test.pdf&encrypt=true
		
		200
		{
		    "presignedUrl": "https://pdf-temp-files.s3-us-west-2.amazonaws.com/28e191b14f3a4f43b16fcef1d53d191e/test.pdf?X-Amz-Expires=900&....",
		    "url": "https://pdf-temp-files.s3-us-west-2.amazonaws.com/28e191b14f3a4f43b16fcef1d53d191e/test.pdf?X-Amz-Expires=....",
		    "error": false,
		    "status": 200,
		    "name": "test.pdf",
		    "remainingCredits": 93574
		}
		
		PUT
		https://pdf-temp-files.s3-us-west-2.amazonaws.com/28e191b14f3a4f43b16fcef1d53d191e/test.pdf?X-Amz-Expires=900&....
		--form 'file=@/path/to/file'
		
		200
		{
		}
		
		Now you can access your file using link from "url" param to from the first step.
		```"
    encrypt: Enable encryption for output file. Must be one of: `true`, `false`.
        name: The name the file will be stored with. Must be a String.
        
    """
    url = f"https://pdf-co2.p.rapidapi.com/v1/file/upload/get-presigned-url"
    querystring = {'encrypt': encrypt, 'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pdf-co2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

