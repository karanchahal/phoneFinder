if [ ! -d dataset ]; then
  unzip -o find_phone_dataset.zip
  mv find_phone dataset/
fi
if [ ! -d Mask_RCNN ]; then
  git clone https://github.com/matterport/Mask_RCNN.git
fi
pip install -r requirements.txt
echo "Setup complete"
