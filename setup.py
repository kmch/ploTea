from setuptools import setup, find_packages

setup(
    name='plotea',
    version='0.0.2',
    packages=find_packages(), # organise the internal dependencies, not external 
    install_requires=[
        # 'abc',
        'autologging',
        # 'cmocean',
        #'matplotlib',
        # 'numpy',
    ],    
    description='Framework for data-rich, interactive, publication-quality plots.',
    author='Kajetan Chrapkiewicz',
    author_email='k.chrapkiewicz17@imperial.ac.uk',
    url='https://github.com/kmch/plotea',
)
