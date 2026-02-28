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