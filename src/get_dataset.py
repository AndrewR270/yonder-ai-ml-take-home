
import os
import truststore
from dotenv import load_dotenv
from roboflow import Roboflow

truststore.inject_into_ssl()
load_dotenv()


api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)
project = rf.workspace("malletbottle2").project("sampled-yd-object-detection")
version = project.version(2)
dataset = version.download("yolov8")
