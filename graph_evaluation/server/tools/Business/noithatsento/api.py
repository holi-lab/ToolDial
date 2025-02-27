import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def noithatsento(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nội thất SENTO là đơn vị chuyên kinh doanh sản phẩm nội thất cho công ty, doanh nghiệp các đại lý trên toàn quốc. SENTO là đơn vị trực tiếp sản xuất các sản phẩm như là:Tủ locker, tủ ngăn kéo, tủ hồ sơ văn phòng, tủ đựng tài liệu, tủ giày dép.
		Các sản phẩm do SENTO sản xuất luôn thích ứng nhanh với thay đổi của thị trường và nhu cầu khách hàng, chúng tôi luôn luôn cải tiến sản phẩm để đưa ra những phẩm phẩm tối ưu nhất, và giá cả cạnh tranh nhất cho khách hàng.
		#noithatsento #tulocke #tủ_hồ_sơ_văn_phòng #tủ_đựng_quần_áo #sanxuattusat #lapdat #tutruonghoc
		Địa chỉ : Kim Chung,Hoài Đức, Hà Nội
		Hotline 0963559655
		Gmail: noithatsentocom@gmail.com
		Google map https://www.google.com/maps?cid=16015397880460403868 
		Website https://noithatsento.com/ 
		https://noithatsento.com/tu-locker 
		https://noithatsento.com/tu-ho-so-van-phong 
		https://noithatsento.com/tu-dung-quan-ao 
		https://www.facebook.com/tusatchinhhang/ 
		https://www.youtube.com/channel/UCyPhGMnrqYBJBEPEo729hMA/about 
		https://noithatsento.blogspot.com/ 
		https://guides.co/p/noithatsento
		https://noithatsento.blogspot.com/
		https://www.misterpoll.com/users/3783283
		https://www.pinterest.com/noithatsentocom
		https://social.technet.microsoft.com/profile/noithatsentocom/
		https://social.msdn.microsoft.com/Profile/noithatsentocom
		https://lightroom.adobe.com/u/noithatsento
		https://www.blogger.com/profile/03420627273045964552
		https://vi.gravatar.com/noithatsentocom
		https://staffmeup.com/profile/noithatsento
		http://www.rohitab.com/discuss/user/531642-noithatsento/
		https://www.behance.net/noithatsento 
		https://www.beatstars.com/noithatsento/about 
		https://issuu.com/noithatsento
		https://profile.hatena.ne.jp/noithatsento/
		https://ok.ru/noithatsento
		https://sites.google.com/view/noithatsento 
		https://gallery.autodesk.com/users/CWQ2L8SEMH8NKF38?relProf=1 
		https://comicvine.gamespot.com/profile/noithatsento/"
    
    """
    url = f"https://noithatsento.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "noithatsento.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

