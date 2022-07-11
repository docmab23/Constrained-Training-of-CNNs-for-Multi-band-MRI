#!/bin/bash

calib=0
help=0


while getopts "ct:s:f:o:i:r:h" OPTION;
do
    case "$OPTION" in
        c) 
        calib=1;;
        t) 
        c_file=${OPTARG};;
        s) 
        slice=${OPTARG};;
        f) 
        filename=${OPTARG};;
        o) 
        calib_dir=${OPTARG};;
        i) 
        in_file=${OPTARG};;
        r) 
        res_dir=${OPTARG};;
        h) 
        help=1;;
    esac
done

echo $calib
echo $c_file
echo $res_dir
echo $in_file
echo $slice
echo $filename

if [[ $help -eq 1 ]];
then 
echo "This is a Bash script to perfrom Slice RAKI and RAKI reconstruction on the Multiband Brain MR data
 -c = if you want to perform training on calibration data 
 -t = The calibration data file in .npy file format
 -s = the slice you want to reconstruct from your undersampled data
 -f = filename for the calibration files
 -o = directory to store calibration data 
 -i = undersampled data to be reconstructed
 -r = Results Directory (to store reconstructed images)
 "
 exit 
fi



if [[ $calib -eq 1 ]];
then 
   command python train_all.py $c_file $slice $filename $calib_dir
fi
#echo "$calib"
command python recon_all.py $in_file $slice $filename $res_dir $calib_dir
