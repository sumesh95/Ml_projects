from setuptools import setup,find_packages

def get_requirments(file_path):
    requirments=[]
    with open(file_path) as fp:
        requirments=fp.readlines()
        requirments=[req.replace("\n","") for req in requirments]

        if '-e .' in requirments:
            requirments.remove('-e .')
    return requirments        



setup(
name="Ml Project",
version=0.01,
author="SUMESH M",
packages=find_packages(),
install_requires=get_requirments('requirements.txt')
    
)