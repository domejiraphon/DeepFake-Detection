import os 
import sys 
import glob 
import argparse 
from loguru import logger 

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="/vast/jy3694/10")
parser.add_argument('--dst', type=str, default="/vast/jy3694/captcha_dataset")
args = parser.parse_args()

def mkdir(path):
  if not os.path.exists(path):
    os.system(f"mkdir -p {path}")

@logger.catch
def main():
  type_path = "fake"
  for i, img_path in enumerate(sorted(os.listdir(args.path))):
   
    person = img_path.split('_')[0]
    challenge = img_path.split('_')[1]
    dst_path = os.path.join(args.dst, challenge, person, type_path.split('/')[-1])
    #if os.path.isfile(dst_path):
    #  os.system(f"rm -f {dst_path}")
    #if str(challenge) not in os.listdir(os.path.join(args.dst, person)):
    #  mkdir(os.path.join(args.dst, challenge, person))
    mkdir(dst_path)
    cmd = f"cp {os.path.join(args.path, img_path)} {dst_path}"
  
    os.system(cmd)

    img_path = os.path.join(dst_path, img_path)
    new_name = img_path.split('_')[2] + '.jpg'
    new_path = os.path.join(dst_path, "%05d" %i+".jpg")
              
    cmd = f"mv {os.path.join(dst_path, img_path)} {new_path}"
    print(cmd)
    os.system(cmd)

@logger.catch
def mai():
  for type_path in sorted(os.listdir(args.path)):
    for i, img_path in enumerate(sorted(os.listdir(os.path.join(args.path, type_path)))):
     
      person = img_path.split('_')[0]
      challenge = img_path.split('_')[1]
      dst_path = os.path.join(args.dst, challenge, person, type_path.split('/')[-1])
      #if os.path.isfile(dst_path):
      #  os.system(f"rm -f {dst_path}")
      #if str(challenge) not in os.listdir(os.path.join(args.dst, person)):
      #  mkdir(os.path.join(args.dst, challenge, person))
      mkdir(dst_path)
      cmd = f"cp {os.path.join(args.path, type_path, img_path)} {dst_path}"
    
      os.system(cmd)

      img_path = os.path.join(dst_path, img_path)
      new_name = img_path.split('_')[2] + '.jpg'
      new_path = os.path.join(dst_path, "%06d" %i+".jpg")
                
      cmd = f"mv {os.path.join(dst_path, img_path)} {new_path}"
      print(cmd)
      os.system(cmd)    
    
      
      


if __name__ == "__main__":
  main()