{
   "general": {
      "demo_title": "An Implementation of Image Retrieval using Scene Graphs",
      "description": "The goal is to implement the method",
      "requirements": "docker",
      "timeout": "3600"
   },
   "build": {
      "url": "git@github.com:titiuo/RR-Image-Retrieval-using-Scene-Graphs.git",
      "rev": "origin/main",
      "dockerfile": ".ipol/Dockerfile"
   },
   "archive": {
      "files": {
         "input_0.csv": "input file",
         "predicted_labels.csv": "output file"
      },
      "params": [
         "image_description"
      ],
      "enable_reconstruct": "true",
      "info": {
         "run_time": "run time"
      }
   },
   "inputs": [
      {
         "description": ".csv file containing at least one textual field",
         "ext": ".csv",
         "type": "data",
         "required": true,
         "max_weight": "10*1024*1024"
      }
   ],
   "params": [
      {
         "id": "image_description",
         "type": "text",
         "label": "Description",
         "comments": "Write here the description of the image",
         "values": {
            "default": ""
         }
      }
   ],
   "results": [
      {
         "type": "file_download",
         "label": "Download clustering output",
         "contents": "predicted_labels.csv"
      },
      {
         "contents": "test.txt",
         "type": "html_file",
         "label": "html"
      },
      {
         "type": "file_download",
         "label": "Fichier txt",
         "contents": "test.txt"
      }
   ],
   "run": "python3 $bin/main.py ${image_description}"
}























######################### Old docker files
# use one of the images from this repository: https://github.com/centreborelli/ipol-docker-images/
FROM registry.ipol.im/ipol:v2-py3.11

# install additional debian packages
#COPY .ipol/packages.txt packages.txt
#RUN apt-get update && apt-get install -y $(cat packages.txt) && rm -rf /var/lib/apt/lists/* && rm packages.txt

# copy the requirements.txt and install python packages
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt && rm requirements.txt

# copy the code to $bin
ENV bin /workdir/bin/
RUN mkdir -p $bin
WORKDIR $bin
COPY . .

# the execution will happen in the folder /workdir/exec
# it will be created by IPOL

# some QoL tweaks
ENV PYTHONDONTWRITEBYTECODE 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
ENV PATH $bin:$PATH

# $HOME is writable by the user `ipol`, but 
ENV HOME /home/ipol
# chmod 777 so that any user can use the HOME, in case the docker is run with -u 1001:1001
RUN groupadd -g 1000 ipol && useradd -m -u 1000 -g 1000 ipol -d $HOME && chmod -R 777 $HOME
USER ipol