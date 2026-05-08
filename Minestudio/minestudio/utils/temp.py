import os
import logging

def get_mine_studio_dir():
    """
    获取 MineStudio 的目录。如果环境变量 MINESTUDIO_DIR 存在，则使用该目录；
    否则，在用户缓存目录中创建一个 MineStudio 专属子目录。
    """
    mine_studio_dir = os.getenv("MINESTUDIO_DIR")
    if mine_studio_dir:
        logging.info(f"Detecting MINESTUDIO_DIR: {mine_studio_dir}")
    else:
        base_cache_dir = os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
        mine_studio_dir = os.path.join(base_cache_dir, "MineStudio")
        logging.info(f"Environment variable MINESTUDIO_DIR is missing，using default cache directory: {mine_studio_dir}")
    
    os.makedirs(mine_studio_dir, exist_ok=True)
    return mine_studio_dir

if __name__ == '__main__':
    mine_studio_dir = get_mine_studio_dir()
    print(f"MineStudio 最终目录是: {mine_studio_dir}")
