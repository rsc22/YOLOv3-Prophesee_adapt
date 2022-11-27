import os
import sys

this_folder = os.path.dirname( __file__ )
os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))


def explore_folder(folder):
    if folder.split('\\')[-1][0] == '.':
        return False
    elif this_folder in folder:
        return False
    elif '.py' in folder or '.txt' in folder:
        apply_change(folder, '.py' if '.py' in folder else '.txt')
        return True
    else:
        try:
            dirs = os.listdir(folder)
            for dir in dirs:
                explore_folder(os.path.join(folder, dir))
        except Exception as e:
            print('Invalid file')
            return False
        return True

def apply_change(file, extension):
    new_file = file.replace(extension, '_new'+extension)
    with open(file, 'r') as original:
        with open(new_file, 'w') as new:
            for line in original.readlines():
                new.write(line.replace('', ''))
    os.remove(file)
    os.rename(new_file, file)

if __name__ == '__main__':
    explore_folder(r'F:\Documentos\Data_Science\Master\UOC_Ingenieria_Computacional_y_Matematica\Master_UOC\TFM\YOLO\PyTorch-YOLOv3-kitti')