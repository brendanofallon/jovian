############ TARGET ############
FROM --platform=linux/amd64 pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEFAULT_USERNAME=user
RUN useradd $DEFAULT_USERNAME --create-home --user-group
ENV HOME /home/$DEFAULT_USERNAME

WORKDIR $HOME


RUN apt-get update && apt-get -y install gcc

USER $DEFAULT_USERNAME

ENV PATH $PATH:$HOME/.local/bin

COPY pip_requirements.txt .
RUN pip install -r pip_requirements.txt

COPY . $HOME/dnaseq2seq

RUN python dnaseq2seq/dnaseq2seq/main.py -h
ENTRYPOINT ["python", "dnaseq2seq/dnaseq2seq/main.py"]
