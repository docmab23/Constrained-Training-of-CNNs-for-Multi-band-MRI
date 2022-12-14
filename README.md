# SMS-RAKI
RAKI reconstruction for simultaneous multi-slice imaging

# Instructions: 

1. git clone https://github.com/randomprogram/SMS-RAKI --branch mb_ipat_pipeline 
2. pip install -r requirements.txt
3. bash mb.sh -h (to understand the usage of the bash script).
 •	 -c = if you want to perform training on calibration data 
 •	 -t = The calibration data file.
 •	 -s = the slice you want to reconstruct from your undersampled data
 •	 -f = filename for the calibration files
 •	 -o = directory to store calibration data 
 •	 -i = undersampled data to be reconstructed
 •	 -r = Results Directory (to store the reconstructed images)
 
 # Run as follows if you want to perfrom training along with reconstrcution on undersampled data:
 bash mb.sh -c -t "calib_file" -s "slice no" -f "filename" 
 -o "output directory for trained models"
 -i "input volume which needs to be reconstructed"
 -r "results directory"
  
 # If you are on Cn2l server follow these steps:
    •	ssh node15
    •	cd Mannan
    •	conda activate slice 
    •	bash mb.sh -c -t <calib_file>  -s <slice no.>  -f <filename> -o <output directory for trained models> -i <input volume which needs to be reconstructed> -r <results directory>

  
 
![image](https://user-images.githubusercontent.com/31487695/207736243-692a810b-819c-4d82-85d2-8d8ba399567a.png)
