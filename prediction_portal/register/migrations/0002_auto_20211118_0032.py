# Generated by Django 3.1.7 on 2021-11-17 19:02

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('register', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='register',
            name='age',
        ),
        migrations.RemoveField(
            model_name='register',
            name='first_name',
        ),
        migrations.RemoveField(
            model_name='register',
            name='last_name',
        ),
        migrations.RemoveField(
            model_name='register',
            name='phone_number',
        ),
    ]