cd ../

jupyter-notebook \
  --allow-root \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --NotebookApp.base_url='/JF-8089' \
  --NotebookApp.ip='0.0.0.0' \
  --NotebookApp.port=8888 \
  --no-browser
