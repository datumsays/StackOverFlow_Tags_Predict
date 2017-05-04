from django.forms import ModelForm, Textarea
from .models import Posts

class InputForm(ModelForm):

    class Meta:
        model  = Posts
        fields = ['content']
        widgets = {
            'content': Textarea(attrs={'cols': 48, 'rows': 15})
        }