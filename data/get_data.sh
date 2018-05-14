#!/bin/bash

export ALL=false;
while true ; do
  case "$1" in
    --db) export DB="$2" ;export ALL=false; shift 2 ;;
    --all) export ALL=true ; shift 1 ;;
    *) break ;;
  esac
done

#Stanford Background dataset
if [[ ( "$ALL" == true ) || ( "$DB" == 'background' ) ]];
then
    wget -qO- "http://dags.stanford.edu/data/iccv09Data.tar.gz" | tar xz -C .
fi

#SUN DB (reduced)
if [[ ( "$ALL" == true ) || ( "$DB" == 'sun' ) ]];
then
    wget "http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz"
    tar -xzf SUN2012.tar.gz
    rm SUN2012.tar.gz
    python3 clean_sun.py
fi

#SUN DB (FULL)

if [[ ( "$ALL" == true ) || ( "$DB" == 'sun_full' ) ]];
then
    wget "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    tar -xzf SUN397.tar.gz
    rm SUN397.tar.gz
    python3 clean_sun.py --full
fi

#PUBFIG DB
if [[ ( "$ALL" == true ) || ( "$DB" == 'pubfig' ) ]];
then
	mkdir pubfig
	wget -O "pubfig/dev_urls.txt" "http://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_urls.txt"
	python3 parse_pubfig.py

	while IFS= read -r col1 col2
	do
	    wget -t 2 -T 4 -O pubfig/$col1 $col2
	done < pubfig/dev_urls_parsed.txt
	rm pubfig/dev_urls.txt
	rm pubfig/dev_urls_parsed.txt
	python3 clean_pubfig.py
fi

#CAL256
if [[ ( "$ALL" == true ) || ( "$DB" == 'cal256' ) ]];
then
    	wget "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
	    tar 256_ObjectCategories.tar
fi

#SiftFlow

if [[ ( "$ALL" == true ) || ( "$DB" == 'siftflow' ) ]];
then
    	wget "http://www.cs.unc.edu/~jtighe/Papers/ECCV10/siftflow/SiftFlowDataset.zip"
	    unzip SiftFlowDataset.zip
      rm SiftFlowDataset.zip
fi

#LFW

if [[ ( "$ALL" == true ) || ( "$DB" == 'lfw' ) ]];
then
    	wget "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
	    tar xzf lfw.tgz
        rm lfw.tgz
        python3 clean_lfw.py
fi
