from django.shortcuts import render,redirect
from django.contrib.auth import login, authenticate, logout
from register.forms import AccountAuthForm
from register.forms import RegistrationForm
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError

# Create your views here.
def registration_view(request):
    context = {}
    if request.POST:
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            email = form.cleaned_data.get('email')
            raw_password = form.cleaned_data.get('password1')
            account = authenticate(email=email, password=raw_password)
            login(request, account)
            return redirect('home')
            
        else:
            if request.POST['password1']!=request.POST['password2']:
                context['registration_form'] = form
                context['error']='Both the passwords should match.'
                return render(request, 'register/register.html', context)
            else: 
                context['registration_form'] = form
                context['error']='Please enter details in the appropriate format.'
                return render(request, 'register/register.html', context)
    else: #GET request
        form = RegistrationForm()
        context['registration_form'] = form
    return render(request, 'register/register.html', context)


def login_view(request):
    context={}
    user=request.user
    if user.is_authenticated:
        return redirect('home')
    if request.POST:
        form=AccountAuthForm(request.POST)
        if form.is_valid():
            email=request.POST['email']
            password=request.POST['password']
            user=authenticate(email=email, password=password)
            
            if user is None:
                context['login_form'] = form
                context['error']='Invalid credentials.'
                return render(request,'register/login.html',context)
            else:
                    login(request,user)
                    return redirect('home')
        else:
            form=AccountAuthForm()
            context['login_form'] = form
            context['error']='Please enter details in the appropriate format.'
            return render(request, 'register/login.html', context)
    else:
        form=AccountAuthForm()
    context['login_form']=form
    return render(request,'register/login.html',context)

@login_required  
def logout_view(request):
    if request.method=='POST':
        logout(request)
        return redirect('home')

