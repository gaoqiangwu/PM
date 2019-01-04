"""
These functions can process data from
github.com/chinese-poetry/chinese-poetry/json to make regulated verse data
ready to use for training.

If you don't have jsons already, just
git clone https://github.com/chinese-poetry/chinese-poetry.git
to get it.

Regulated verse forms are expected to be tuples of (number of couplets,
characters per couplet). For reference:
wujue 五言絕句 = (2, 10)
qijue 七言絕句 = (2, 14)
wulv 五言律詩 = (4, 10)
qilv 七言律詩 = (4, 14)
"""

import glob
import os
import pandas as pd


def unregulated(paragraphs):
    """
    如果df行描述了不受控制的诗句，则返回True。.
    """
    if all(len(couplet) == len(paragraphs[0]) for couplet in paragraphs):
        return False
    else:
        return True


def get_poems_in_df(df, form):
    """
    返回一个txt友好的字符串，只包含指定格式的df格式的诗歌。
    """
    big_string = ""
    for row in range(len(df)):
        if len(df["strains"][row]) != form[0] or \
                len(df["strains"][row][0]) - 2 != form[1]:
            continue
        if "○" in str(df["paragraphs"][row]):
            continue
        if unregulated(df["paragraphs"][row]):
            continue
        big_string += df["title"][row] + ":"
        for couplet in df["paragraphs"][row]:
            big_string += couplet
        big_string += "\n"
    return big_string


def get_poems_in_dir(dir, form, save_dir):
    """
    保存到保存在dir格式的save_dir诗歌在单独的txt文件中.
    """
    files = [f for f in os.listdir(dir) if "poet" in f]
    for file in files:  # Restart partway through if kernel dies.
        with open(os.path.join(save_dir, file[:-5] + ".txt"), "w") as data:
            print("Now reading " + file)
            df = pd.read_json(os.path.join(dir, file))
            poems = get_poems_in_df(df, form)
            data.write(poems)
            print(str(len(poems)) + " chars written to "
                  + save_dir + "/" + file[:-5] + ".txt")
    return 0


def combine_txt(txts_dir, save_file):
    """
   在txts_dir中合并.txt文件并保存为save_file
    """
    read_files = glob.glob(os.path.join(txts_dir, "*.txt"))
    with open(save_file, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())
    return 0
