from setuptools import setup, find_packages                    
                                                                    
setup(                                                         
    name='llm_unichat',                                  
    version='0.1.2',                                           
    author='amidabuddha',                                        
    author_email='your.email@example.com',                     
    description='Universal Python API client for OpenAI, MistralAI, Anthropic, xAI, and Google AI.',         
    long_description=open('README.md').read(),                 
    long_description_content_type='text/markdown',             
    url='https://github.com/amidabuddha/llm-unichat',  
    packages=find_packages(),                                  
    classifiers=[                                              
        'Programming Language :: Python :: 3',                 
        'License :: OSI Approved :: MIT License',              
        'Operating System :: OS Independent',                  
    ],                                                         
    python_requires='>=3.6',                                   
    install_requires=[                                         
        'anthropic~=0.39.0',
        'mistralai~=1.2.3',
        'openai~=1.55.0',
        'google-generativeai~=0.8.3',                                            
    ],                                                         
) 