from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='powder_alert2.0',
      version="0.0.1",
      description="Powder Alert 2.0 Model",
      author="Powder Alert 2.0 team",
      #url="https://github.com/MadMax1995bb/powder_alert2.0",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
