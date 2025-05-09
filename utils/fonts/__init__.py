import os
import shutil

import matplotlib

file_dir = os.path.dirname(os.path.realpath(__file__))
file_names = os.listdir(file_dir)
font_names = [file_name for file_name in file_names if file_name.endswith('.ttf')]

dst_dir = os.path.join(os.path.dirname(matplotlib.matplotlib_fname()), 'fonts/ttf')
font_names_dst = os.listdir(dst_dir)

FONT_MAPPER = {}
updated = []
for font_name in font_names:
    font_pth = os.path.join(file_dir, font_name)
    FONT_MAPPER[font_name.split('.')[0]] = font_pth
    if font_name not in font_names_dst:
        shutil.copy(font_pth, os.path.join(dst_dir, font_name))
        updated.append(font_name)

# if True:
if len(updated) > 0:
    librc_pth = os.path.join(os.path.dirname(matplotlib.matplotlib_fname()), 'matplotlibrc')
    if os.path.exists(librc_pth):
        with open(librc_pth, 'r') as f:
            lines = f.readlines()
        lines_new = []
        for line in lines:
            line = line.replace('#font.serif:', 'font.serif:')
            line = line.replace('#font.sans-serif:', 'font.sans-serif:')
            lines_new.append(line + os.linesep)
        with open(librc_pth, 'w') as f:
            f.writelines(lines_new)

    print('Auto install fonts for matplotlib', font_names)
    shutil.rmtree(matplotlib.get_cachedir())
