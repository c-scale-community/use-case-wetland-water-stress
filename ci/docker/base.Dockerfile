FROM continuumio/miniconda3:4.10.3p1

RUN apt-get -q update && apt-get -q -y install eatmydata
RUN eatmydata -- apt-get -q -y install build-essential git python3-opencv

RUN eatmydata -- /opt/conda/bin/conda install --quiet -y python=3.8
RUN eatmydata -- /opt/conda/bin/conda install --channel conda-forge --quiet -y gdal=3.0.2
RUN eatmydata -- /opt/conda/bin/conda install --channel conda-forge --quiet -y cartopy
RUN eatmydata -- /opt/conda/bin/conda install --channel conda-forge --quiet -y numpy
