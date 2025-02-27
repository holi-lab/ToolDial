import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def new88(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "new88top.com - new88  Nhà cái uy tín hàng đầu Châu Á, Link vào new88 chính thức cung cấp các game cá cược như: thể thao, đá gà, casino online...	#nạp tiền new88 #rút tiền new88 #đăng nhập new88 #đăng ký new88 #NEW88
		SDT 0971.97.63.63
		Địa chỉ : 266 P. Minh Khai, Mai Động, Hai Bà Trưng, Hà Nội, Việt Nam
		Gmail: new88top@gmail.com	
		Website https://new88top.com/    
		Xem thêm 
		https://new88top.com/nap-tien-new88/   	
		https://new88top.com/rut-tien-new88/  	
		https://new88top.com/dang-ky-new88/    	
		https://new88top.com/dang-nhap-new88/    		
		https://sites.google.com/view/new88top 	
		https://www.emazoo.com/new88top
		https://justpaste.it/84mvb
		https://justpaste.it/u/new88top
		https://rollbol.com/new88top
		https://anchor.fm/new88top
		https://support.advancedcustomfields.com/forums/users/new88top/
		https://new88top.amebaownd.com/posts/34182313
		https://www.themehorse.com/support-forum/users/new88top
		https://www.figma.com/@new88top
		https://www.joomlart.com/forums/u/new88top
		https://62747fd07a9f3.site123.me/
		https://knowyourmeme.com/users/new88top
		https://gitlab.physics.muni.cz/-/snippets/1766
		https://www.docdroid.net/pLnIzHz/new88top-pdf
		https://reedsy.com/discovery/user/new88top
		https://www.furaffinity.net/user/new88top/
		https://new88top.simdif.com/
		https://fundrazr.com/profiles/new88-top"
    
    """
    url = f"https://new88top.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "new88top.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

