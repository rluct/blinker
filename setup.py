from setuptools import setup

setup(
    name='blinker',
    version='0.1',
    description='OpenAI Gym Wrappers for introducing active, partial observability.',
    url='https://github.com/rluct/blinker',
    author='UCT RL',
    author_email='sebastianbod@gmail.com',
    license='MIT',
    install_requires=['gym'],
    test_suite='nose.collector',
    tests_require=['nose'],
    packages=['blinker'],
    zip_safe=False
)