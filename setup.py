from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='PACKAGENAME',
   version='1.0',
   description='An implementation of game theory of mind in a agent based framework following the implementation of Devaine, et al. (2017).',
   license="MIT",
   long_description=long_description,
   author='Man Foo',
   author_email='foomail@foo.com',
   url="http://www.foopackage.com/",
   packages=['foo'],  #same as name
   install_requires=['bar', 'greek'], #external packages as dependencies
   scripts=[
            'scripts/cool',
            'scripts/skype',
           ]
)