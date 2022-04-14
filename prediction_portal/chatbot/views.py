from django.shortcuts import render
import requests, json
from django.http import JsonResponse
from chatbot.chat import get_response
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
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
    response = get_response(msg)
    message = {"answer": response}
    json_mess = JsonResponse(message)
    print(response)
    return json_mess


# Create your views here.
