#!/usr/bin/env bash
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get dist-upgrade -y
wget http://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh
bash Anaconda3-4.3.0-Linux-x86_64.sh -b
export PATH=/home/ubuntu/anaconda3/bin:$PATH
sudo apt-get install build-essential -y
conda install -c anaconda qt=4.8.6 -y
pip install -r requirements.txt -y

python -m nltk.downloader punkt
python - nltk.downloader stopwords
python -m nltk.downloader wordnet