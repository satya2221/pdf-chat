# Generated by Django 5.2 on 2025-04-22 05:40

import core.utils
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.CharField(default=core.utils.generate_id, max_length=200, primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('is_deleted', models.BooleanField(default=False)),
                ('name', models.CharField(max_length=200)),
                ('file', models.FileField(upload_to='documents/')),
                ('raw_text', models.TextField(blank=True, null=True)),
                ('summary', models.TextField(blank=True, null=True)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('processing', 'Processing'), ('complete', 'Complete')], default='pending', max_length=200)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
