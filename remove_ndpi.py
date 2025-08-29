import time
import glob
import shutil
import schedule
import subprocess
from pathlib import Path


Path("/workspace/data/bcc-projekt-digital/bcc").mkdir(parents=True, exist_ok=True)

def remove_ndpi():
    print('Watching for .ndpi files in bcc folder ...')
    slides_in_bcc = glob.glob('/workspace/data/bcc-projekt-digital/bcc/*')
    tiled_WSIs = glob.glob('/workspace/data/cv_methods/tmi2022/WSI/bcc/pyramid/data/*')
    slides_in_bcc = [s.split('/')[-1].split('.')[0] for s in slides_in_bcc]
    tiled_WSIs = [s.split('/')[-1] for s in tiled_WSIs]
    r=0
    for slide in slides_in_bcc:
        if set([slide]).issubset(tiled_WSIs):
            r+=1
            subprocess.run(['rm','-f', '/workspace/data/bcc-projekt-digital/bcc/' + slide + '.ndpi'])
    if r!=0:
        print('Removed', r, 'ndpis')
schedule.every(1).minutes.do(remove_ndpi)

while True:
    schedule.run_pending()
    time.sleep(1)
