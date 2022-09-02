############ TARGET ############
FROM --platform=linux/amd64 python:3.10

ENV DEFAULT_USERNAME=user
RUN useradd $DEFAULT_USERNAME --create-home --user-group
ENV HOME /home/$DEFAULT_USERNAME

WORKDIR $HOME

USER $DEFAULT_USERNAME


RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
RUN chmod u+x Miniconda3-py38_4.12.0-Linux-x86_64.sh
RUN ./Miniconda3-py38_4.12.0-Linux-x86_64.sh -b

ENV PATH $PATH:$HOME/.local/bin
ENV PATH $HOME/miniconda3/bin/:$PATH

COPY . $HOME/dnaseq2seq


#RUN chown -R $DEFAULT_USERNAME ./dnaseq2seq/
#RUN chown -R $DEFAULT_USERNAME ./miniconda3/


RUN pip install $HOME/dnaseq2seq

ENV JOVIAN_MODEL $HOME/dnaseq2seq/wgs_10m_halfhuge_rep2_cont3_epoch12.model
ENV JOVIAN_CLASSIFIER $HOME/dnaseq2seq/varmerge.model

RUN python dnaseq2seq/dnaseq2seq/main.py -h
ENTRYPOINT ["python", "dnaseq2seq/dnaseq2seq/main.py"]
