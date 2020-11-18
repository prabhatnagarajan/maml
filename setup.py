import codecs
from setuptools import find_packages
from setuptools import setup


install_requires = [
    'torch>=1.3.0',
    'numpy',
    'matplotlib',
]

setup(name='maml',
      version='0.0.0',
      description='maml, an Implementation of Model Agnostic Meta Learning',
      long_description=codecs.open('README.md', 'r', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      author='Prabhat Nagarajan',
      author_email='prabhat@prabhatnagarajan.com',
      license='MIT License',
      packages=find_packages(),
      install_requires=install_requires,
      test_requires=test_requires
      )