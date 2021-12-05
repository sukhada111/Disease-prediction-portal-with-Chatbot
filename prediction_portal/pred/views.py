from django.shortcuts import render
import requests

def pred(request):
    return render(request, 'predic.html')