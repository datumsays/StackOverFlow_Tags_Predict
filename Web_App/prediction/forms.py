from django import forms
from .models import Posts

class InputForm(forms.ModelForm):

    class Meta:
        model  = Posts
        fields = ['content']