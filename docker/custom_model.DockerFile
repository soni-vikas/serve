FROM pytorch/torchserve:latest-gpu

USER root

RUN apt-get update -y && apt-get install apt-file -y && apt-file update && apt-get install -y python3-dev build-essential
RUN apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
RUN apt-get update -y && apt-get install -y docker-ce docker-ce-cli containerd.io

COPY examples/bert/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY run.sh run.sh

RUN chmod 700 run.sh
CMD ["run.sh"]
ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["/bin/bash", "-c", "torchserve", "--start", "--model-store", "/model_store", "&&", "tail", "-f", "/dev/null"]

#ENTRYPOINT ["torchserve", "--start", "--model-store", "/model_store"]

#    1  torchserve --start --model-store ~/model-store --models ~/model-store/custom_bert.mar
#    2  cat /tmp/a
#    3  touch /tmp/a
#    4  vim /tmp/a
#    5  pip install -r model-store/requirements.txt
#    6  top
#    7  clear
#    8  torchserve --stop
#    9  torchserve --start --model-store ~/model-store --models ~/model-store/custom_bert.mar
#   10  top
#   11  torchserve --stop
#   12  torchserve --start --model-store ~/model-store --models ~/model-store/custom_bert.mar
#   13  torchserve --stop
#   14  torchserve --stop && torchserve --start --model-store ~/model-store --models ~/model-store/custom_bert.mar
#   15  torchserve --stop && torchserve --start --model-store ~/model-store
#   16  tail -f /tmp/log*
#   17  find  / | grep jsr
#   18  find  / | grep jar
#   19  torchserve --stop && torchserve --start --model-store ~/model-store
#   20  grep -ir resource /
#   21  grep -ir resource / | grep resource
#   22  grep -ir resource ~/
#   23  clear
#   24  torchserve --stop && torchserve --start --model-store ~/model-sto
#   25  torchserve --stop && torchserve --start --model-store ~/model-store
#   26  g
#   27  git clone
#   28  apt install git
#   29  torchserve --stop && torchserve --start --model-store ~/model-store
#   30  torchserve --stop
#   31  torchserve --stop && torchserve --start --model-store ~/model-store
#   32  torchserve --stop
#   33  torchserve --stop && torchserve --start --model-store ~/model-store
#   34  torchserve --stop