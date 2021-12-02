from django.shortcuts import render

def medicine_info(request):
    return render(request, 'medicine_info/medicine_info.html')