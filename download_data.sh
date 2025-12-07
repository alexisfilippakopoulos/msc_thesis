# This script explains how to download M-SENA datasets
# Usage: bash download_data.sh
# and stores the data to the path you currently are
# author: efthymisgeo

###############################################################################
#### Download MOSEI
###############################################################################
mkdir mosei
mkdir processed
cd processed
gdown <aligned>
gdown <unaligned>
### for large files
### 1.Copy link
### 2.New browser window
### 3.Paste
### 4.Change 2 https://drive.google.com/uc?id=1LKsuRkO0MqfdLZIyjoBgEEQPH8napiKi
### 5.F12
### 6.Download + Cancel download
### 7.Take 200 adress on the left and copy as cURL
### 8.Direct paste on terminal and "> Raw.zip"

###############################################################################
#### Download MOSI
###############################################################################
mkdir processed
cd processed
gdown 1VqjkYqcgUlggZVN7B3NpXwQ-BIWIXxkj
gdown 1U_RpJB_PV-JRgCs694UHczLr2h5YJOma
cd ..
gdown 1dqyQI8iOHGjwofgq1DvAD-inW-zrTa2h
gdown 1-38kUvB-3LTmMvb1bgkSVxgup0UvYQUp
cd ..
###############################################################################
#### Download SIMS
###############################################################################
mkdir sims
cd sims
mkdir processed
cd processed
 
