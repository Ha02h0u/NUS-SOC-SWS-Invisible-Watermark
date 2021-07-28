from att import *
import sys
import os
import time

datadir = './data/'
savedir = './save/'
imgs = os.listdir(datadir)

def usage():
    print('''
Usage: python3 mani.py [options] [source_video] [dest_video]
    options: cut_height/cut_width/resize/bright/rotation/shelter/salt_pepper
    example: python3 mani.py cut_height ./watermark.mp4 ./attacked.mp4
    ''')
    exit(0)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage()
    to_image = 'ffmpeg -i ' + sys.argv[2] + ' -f image2 -vf fps=fps=20 -qscale:v 2 '+datadir+'tmp-%05d.jpeg'
    to_video = 'ffmpeg -f image2 -i '+savedir+'tmp-%05d-out.jpeg -vcodec libx264 -r 20 ' + sys.argv[3]
    start = time.time()

    attack = {
        'cut_height': cut_att_height,
        'cut_width': cut_att_width,
        'resize': resize_att,
        'bright': bright_att,
        'shelter': shelter_att,
        'salt_pepper': salt_pepper_att,
        'rotation': rot_att
    }

    if not os.path.exists('./save'):
        os.mkdir('./save')
    if not os.path.exists('./data'):
        os.mkdir('./data')
    os.system('rm -r ./data; rm -r ./save; mkdir data; mkdir save')
    os.system(to_image)
    print("Processing...")
    for img in imgs:
        # an annoying stuff which may leads to failure when running on MacOS
        if img == 'DS.Store':
            continue
        output = img[:-5] + '-out.jpeg'
        output = os.path.join(savedir,output)
        img = os.path.join(datadir, img)
        try:
            attack.get(sys.argv[1])(img, output)
        except :
            print("The attack failed. You may typed wrong.")
            exit(1)

    os.system(to_video)
    print("Succeed!")

    end = time.time()
    print("Time used: %.2f s" % (end - start))
    
