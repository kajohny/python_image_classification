# Generated by Django 3.2.10 on 2022-03-03 18:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classifierApp', '0003_alter_prediction_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prediction',
            name='image',
            field=models.TextField(blank=True, null=True),
        ),
    ]
