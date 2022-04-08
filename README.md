# dxw-wordcloud
A dxw plugin to generate wordclouds based on dx projects


# Development setup

Install dependencies using (use a conda environment and activate it beforehand):

    $pip install -r requirements.txt

To run the script locally use:

    $python dxwc/dx_wc.py  --dx <path_to_a_dx_project_folder> 


Package the python script as a stand-alone distributable (with dependencies) 

    $pyinstaller dx_wc.spec

TODO: explain why we use spec file, we need to add additional files: wordcloud and nltk

Link the plugin in dxw:

    $dxw plugin link

Link the distribution of the python module in the install folder for dxw command

    $ln -s ~/Work/dxw-wordcloud/dist/dx_wc/ ~/.dxw/dx_wc/0.0.1

Now you can run the plugin from dxw:

    $dxw wc --dx . --max-words 10 --output +wc3