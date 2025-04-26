import os.path
import shutil

__all__ = []


def move_file():
    root = 'D:\Datasets\ADE20K'
    proc_folder = 'training'
    image_folder = 'images'
    png_folder = 'annotations'
    json_folder = 'jsons'
    part_folder = 'instance'
    for folder in [image_folder, png_folder, json_folder, part_folder]:
        folder_dir = os.path.join(root, folder, proc_folder)
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

    counts = 0
    proc_dir = os.path.join(root, proc_folder)
    for sub1_name in os.listdir(proc_dir):
        sub1_dir = os.path.join(proc_dir, sub1_name)
        for sub2_name in os.listdir(sub1_dir):
            sub2_dir = os.path.join(sub1_dir, sub2_name)
            for file_name in os.listdir(sub2_dir):
                file_pth = os.path.join(sub2_dir, file_name)
                if os.path.isdir(file_pth):
                    folder = part_folder
                elif '.png' in file_pth:
                    folder = png_folder
                elif '.json' in file_pth:
                    folder = json_folder
                elif '.jpg' in file_pth:
                    folder = image_folder
                    counts += 1
                else:
                    raise Exception('err ' + file_pth)
                file_pth_dst = os.path.join(os.path.join(root, folder, proc_folder, file_name))
                print(file_pth, '->', file_pth_dst)
                shutil.move(file_pth, file_pth_dst)
    print(counts)
    return None


if __name__ == '__main__':
    root = 'D:\Datasets\ADE20K'
    png_folder = 'annotations'
    proc_folder = 'validation'
    set_dir = os.path.join(root, png_folder, proc_folder)
    for file_name in os.listdir(set_dir):
        file_pth = os.path.join(set_dir, file_name)
        file_pth_dst = os.path.join(set_dir, file_name.replace('_seg.png', '.png'))
        if not file_pth == file_pth_dst:
            print(file_pth, '->', file_pth_dst)
            os.rename(file_pth, file_pth_dst)
