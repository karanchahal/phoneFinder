if [ ! -d dataset ]; then
  # Control will enter here if directory 'build' doesn't exist.
  unzip -o find_phone_dataset.zip
  mv find_phone dataset/
fi
if [ ! -d Mask_RCNN ]; then
  git clone https://github.com/matterport/Mask_RCNN.git
fi
pip install scikit-image==0.14.2
echo "Setup complete"
