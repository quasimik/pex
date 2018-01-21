import sys
import os
import time
import json
import numpy as np
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler



class JSONHandler(PatternMatchingEventHandler):
  patterns = ["*.json"]

  def process(self, event):
    """
    event.event_type 
        'modified' | 'created' | 'moved' | 'deleted'
    event.is_directory
        True | False
    event.src_path
        path/to/observed/file
    """

    strip_indices = np.r_[0:3,42:54]
    print event.src_path
    data = json.load(open(event.src_path))
    if data["people"] and data["people"][0] and data["people"][0]["pose_keypoints"]:
      data = np.array(data["people"][0]["pose_keypoints"])
      data = np.delete(data, strip_indices)
      with open("../outfile.json", "a") as of:
        if os.stat("../outfile.json").st_size > 0:
          of.write(",\n")
        json.dump(data.tolist(), of)
        of.close()

    os.remove(event.src_path)

  def on_modified(self, event):
    self.process(event)

if __name__ == '__main__':
  args = sys.argv[1:]
    
  observer = Observer()
  observer.schedule(JSONHandler(), path=args[0] if args else '.')
  observer.start()

  try:
    while True:
      time.sleep(10)
  except KeyboardInterrrupt:
    observer.stop()
    print "Terminating..."
    sys.exit(0)

  observer.join()
