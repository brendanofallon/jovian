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


COPY 100M_s28_cont_mapsus_lolr2_epoch2.model $APPDIR/jenever_model.pt
COPY s28ce40_bamfix.model $APPDIR/classifier.model

ENV JENEVER_MODEL=jenever_model.pt
ENV JENEVER_CLASSIFIER=classifier.model

COPY . $APPDIR
COPY docker-entrypoint.sh /docker-entrypoint.sh

RUN python $APPDIR/dnaseq2seq/main.py -h
#ENTRYPOINT ["ls", "-lhrst"]
ENTRYPOINT ["/docker-entrypoint.sh"]
