from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate
from register.models import Register

class RegistrationForm(UserCreationForm):
        email=forms.EmailField(max_length=100,help_text='Required. Add a valid email address')
        class Meta:
            model=Register
            fields=['email','username','password1','password2','name','phone_number','age','gender']

class AccountAuthForm(forms.ModelForm):
    password=forms.CharField(label='password',widget=forms.PasswordInput)
    
    class Meta:
        model=Register
        fields=['email','password']

    def clean(self):
        email=self.cleaned_data['email']
        password=self.cleaned_data['password']
        # if not authenticate(email=email,password=password):
        #     raise forms.ValidationError("Invalid credentials.")