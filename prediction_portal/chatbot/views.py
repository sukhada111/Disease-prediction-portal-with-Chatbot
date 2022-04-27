from django.shortcuts import render
import requests, json, random
from django.http import JsonResponse
from chatbot.chat import get_response
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required


@csrf_exempt
@login_required
def chat_page(request):
    return render(request, 'chatbot.html')


@csrf_exempt
def chat(request):
    text={'message': "Hello"}
    try:
        text = json.loads(request.body)
        print(text)
        
    except json.decoder.JSONDecodeError:
        print("String could not be converted to JSON")
    msg = text['message']
    response, tag = get_response(msg)
    #Info tags
    infotags=["anxiety relief", "stress relief", "depression relief"]
    url=""
    ar={
        "stress relief":["https://www.youtube.com/watch?v=U7g3Cciwxcc","https://www.youtube.com/watch?v=PNiLq-cZUBU"],
        "anxiety relief":[" https://www.youtube.com/watch?v=79kpoGF8KWU", "https://www.youtube.com/watch?v=AImuCtIokl0","https://www.youtube.com/watch?v=Y6NP9P6BL8A"],
        "depression relief":["https://www.youtube.com/watch?v=1ZYbU82GVz4&ab_channel=SoothingRelaxation","https://www.youtube.com/watch?v=9kzMf5kkQKg", "https://www.youtube.com/watch?v=79kpoGF8KWU"]
    }
    
    if tag in infotags:
        url=random.choice(ar[tag])


    message = {"answer": response}
    json_mess = JsonResponse(message)
    print(message)
    print(response)
    print(url)

    return json_mess


# Create your views here.