import time
import glob
import shutil
import schedule
import subprocess
from pathlib import Path

Path("/workspace/data/bcc-projekt-digital/bcc").mkdir(parents=True, exist_ok=True)

def move_2_BCC_and_tile():
    uploaded_files = glob.glob('/workspace/data/bcc-projekt-digital/*.ndpi')
    if len(uploaded_files)!=0:
        shutil.move(uploaded_files[0], '/workspace/data/bcc-projekt-digital/bcc/')
    else:
        print('There are no files to move and tile')
    print('Moved files.', len(uploaded_files), 'left to move')
    
    slides_in_bcc = glob.glob('/workspace/data/bcc-projekt-digital/bcc/*')
    tiled_WSIs = glob.glob('/workspace/data/cv_methods/tmi2022/WSI/bcc/pyramid/data/*')
    
    if len(slides_in_bcc) != 0:
        subprocess.run('python deepzoom_tiler.py -m 2 3 -b 40 -v ndpi -j 32 --dataset bcc'.split(' ')) # careful for extraspaces
    
    time.sleep(0.2)
    slides_in_bcc = [s.split('/')[-1].split('.')[0] for s in slides_in_bcc]
    tiled_WSIs    = [s.split('/')[-1].split('.')[0] for s in tiled_WSIs]
    

schedule.every(0.05).minutes.do(move_2_BCC_and_tile)
  
while True:
    schedule.run_pending()
    time.sleep(0.01)