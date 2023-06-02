from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='unchecker',
    version='0.1',
    description='Your package description',
    author='Jan Bauer',
    author_email='japhba@users.noreply.github.com',
    url='https://github.com/japhba/unchecker',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
    'console_scripts': [
        'unchecker = unchecker.cli:main'
    ]
}
)
