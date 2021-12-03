from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'index.html')

def aboutUs(request):
    return render(request, 'aboutUs.html')