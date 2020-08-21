import os

for name in os.listdir():
    new_name = name.replace(" ",'')
    os.rename(name, new_name)
