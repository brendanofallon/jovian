############ TARGET ############
FROM --platform=linux/amd64 pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEFAULT_USERNAME=user
RUN useradd $DEFAULT_USERNAME --create-home --user-group
ENV HOME /home/$DEFAULT_USERNAME

ENV APPDIR /jenever

WORKDIR $HOME

RUN apt-get update && apt-get -y install gcc vim coreutils

RUN conda install -c conda-forge scikit-bio

COPY pip_requirements.txt .
RUN /opt/conda/bin/pip install -r pip_requirements.txt

USER $DEFAULT_USERNAME


ENV PYTHONPATH $PYTHONPATH:/opt/conda/lib/python3.10/:

ENV PATH $PATH:$HOME/.local/bin


COPY models/100M_s28_cont_mapsus_lolr2_epoch2.model $APPDIR/100M_s28_cont_mapsus_lolr2_epoch2.model
COPY models/s28ce40_bamfix.model $APPDIR/s28ce40_bamfix.model

ENV JENEVER_MODEL=100M_s28_cont_mapsus_lolr2_epoch2.model
ENV JENEVER_CLASSIFIER=s28ce40_bamfix.model

COPY dnaseq2seq $APPDIR
COPY docker-entrypoint.sh /docker-entrypoint.sh

RUN python $APPDIR/dnaseq2seq/main.py -h
#ENTRYPOINT ["ls", "-lhrst"]
ENTRYPOINT ["/docker-entrypoint.sh"]
