from django.shortcuts import render,redirect,get_object_or_404
from django.contrib.auth.decorators import login_required
import requests

def medicine_info(request):
    return render(request, 'medicine_info/medicine_info.html')