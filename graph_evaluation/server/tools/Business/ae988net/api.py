import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ae988net(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ae988.net ☑️ Nhà cái đá gà cựa sắt uy tín số 1 Việt Nam, đến với ae988 bạn sẽ được xem các trận đá gà cựa sắt, cựa dao cùng các game như: bóng đá, đá gà, casino...	
		#đăng ký ae988 #nạp tiền ae988 #rút tiền ae988 #rút tiền ae988 #tải ứng dụng ae988 #linkvaonhacaiae9882022
		/m/033_l8  /m/033_l8 #Casino 
		Hotline 0382.977.144
		Địa chỉ : 549a Trần Xuân Soạn, Tân Hưng, Quận 7, Thành phố Hồ Chí Minh, Việt Nam
		Gmail: ae988net@gmail.com
		Website https://ae988.net/ 	
		https://ae988.net/dang-ky-ae988/ 	
		https://ae988.net/nap-tien-ae988/ 	  
		https://ae988.net/rut-tien-ae988/ 	
		https://ae988.net/huong-dan-tai-ung-dung-ae988/ 	
		https://en.wikipedia.org/wiki/Online_casino   
		https://www.question2answer.org/qa/user/ae988net
		https://themehunt.com/profile/ae988net
		https://hashnode.com/@ae988net
		https://experiment.com/users/aae988net
		https://comicvine.gamespot.com/profile/ae988net/
		https://gifyu.com/ae988net
		https://www.folkd.com/user/ae988net
		https://sketchfab.com/ae988net
		http://www.wikidot.com/user:info/ae988net
		https://bikepgh.org/message-board/users/ae988net/
		https://gotartwork.com/Profile/ae988net-yeuanh12/144575/
		https://stocktwits.com/ae988net
		https://connect.garmin.com/modern/profile/cb606a6b-6f54-44e3-9428-eb25e8581c3a
		https://ko-fi.com/ae988net43344
		https://linktr.ee/ae988net
		https://about.me/ae988net/
		https://gallery.autodesk.com/users/BJ52FJLKQZLPBMK4
		http://uid.me/ae988net_anhem
		https://gitlab.manjaro.org/ae988net
		https://gitlab.pasteur.fr/ae988net
		https://repo.getmonero.org/ae988net
		https://career.habr.com/ae988net
		https://gitlab.haskell.org/ae988net
		https://cartoonmovement.com/cartoonist/18446
		https://www.pinterest.com/ae988n/_saved/
		https://www.behance.net/ae988net
		https://vi.gravatar.com/ae988net
		https://www.jigsawplanet.com/ae988net?viewas=07c4d0e66612
		https://binaryoptionrobotinfo.com/forums/users/ae988net
		https://www.divephotoguide.com/user/ae988net
		https://mootools.net/forge/profile/ae988net
		https://fr.ulule.com/ae988net/"
    
    """
    url = f"https://ae988net.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ae988net.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

