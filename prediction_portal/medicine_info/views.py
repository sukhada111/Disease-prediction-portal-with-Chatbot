from django.shortcuts import render,redirect,get_object_or_404
from django.contrib.auth.decorators import login_required
import requests

from bs4 import BeautifulSoup
import re

def medicine_info(request):

    # Inp=input("Enter the Medicine Name: ")
    if request.POST:
        Search=[]
        class Gsearch_python:
            def __init__(self,name_search):
                    self.name = name_search
            def Gsearch(self):
                    count = 0
                    try :
                        from googlesearch import search as gsearch
                    except ImportError:
                        print("No Module named 'google' Found")
                    for i in gsearch(self.name, lang='en'):
                        Search.append(i)
        Inp=request.POST['med']
        print(Inp)
        SEQS="netmeds"+str(Inp)
        gs = Gsearch_python(Inp)
        gs.Gsearch()
        URL=""

        sample = "https://www.netmeds.com/prescriptions/"
        for string in Search:
            if (sample in string):
                y = "^" + sample
                x = re.search(y, string)
                if x :
                    URL=string
                    break
                else :
                    continue
        else:
            print("Please Enter the medicine name correctly")

        heads={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9", "Accept-Encoding": "gzip, deflate, br","DNT":"1","Connection":"close","Upgrade-Insecure-Requests": "1"}
        if(len(URL)!=0):
            page=requests.get(URL,headers=heads)
            Soup1=BeautifulSoup(page.content,"html.parser")
            result1 = Soup1.find(id='np_tab1')
            
            result5 = Soup1.find(id='np_tab5')
            
            result7 = Soup1.find('div',class_="right-block")

            Uses = result1.find_all('div', class_='inner-content')
            
            Side_Effects = result5.find_all('div', class_='inner-content')
        
            alternatives =result7.find_all('div', class_='info')

            alt=[]
            uses=[]
            side_Effects=[]
    

            for i in alternatives:
                if "[" not in i.get_text():
                    alt.append(i.get_text().strip().replace("\xa0"," "))
            for i in Uses:
                if "[" not in i.get_text():
                    uses.append(i.get_text().strip().replace("\xa0"," "))
            for i in Side_Effects:
                if "[" not in i.get_text():
                    side_Effects.append(i.get_text().strip().replace("\xa0"," "))
            
            print(alt)
            print(uses)
            print(side_Effects)
            alt=alt[:5]
            # uses=uses[:5]
            use=uses[0].split('?')
            uses=use[1]
            use_list=uses
            if(len(uses)>250):
                use_list=[]
                l4=uses[0][:250]
                x=l4.rindex(',')

                use_list.append(uses[:x])
            # side_Effects=side_Effects[:10]
            s=side_Effects[0][0].upper()
            f=side_Effects[0][0]
            side_Effects[0]=side_Effects[0].replace(f,s,1)
            l2=side_Effects
            if(len(side_Effects[0])>250):
                l2=[]
                l3=side_Effects[0][:250]
                x=l3.rindex(',')

                l2.append(side_Effects[0][:x])
            context={}
            context['alt']=alt
            context['uses']=uses
            context['side']=l2
            context['input']=Inp
            
            l1=[]
            l1.append(context)

            return render(request, 'medicine_info/medicine_info.html',{'l1':l1})
        else:
            return render(request, 'medicine_info/medicine_info.html',{'error':"Medicine not found."})
    
    else:
        return render(request, 'medicine_info/medicine_info.html')
