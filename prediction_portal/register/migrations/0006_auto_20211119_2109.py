# Generated by Django 3.1.7 on 2021-11-19 15:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('register', '0005_register_gender'),
    ]

    operations = [
        migrations.AlterField(
            model_name='register',
            name='gender',
            field=models.CharField(max_length=20, null=True),
        ),
    ]
