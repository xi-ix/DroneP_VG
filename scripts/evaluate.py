import os

def get_filenames(directory_path):
    """
    读取指定目录下的所有文件名并返回列表
    """
    try:
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except FileNotFoundError:
        return "错误：指定的路径不存在"
    except PermissionError:
        return "错误：没有权限访问该目录"

def load_data_GT(file_path):
    """
    [x1, y1, w, h, confidence] 形式的 Ground Truth 文件
    返回嵌套列表，例如: [[0, 120, 30, 50], [1, 45, 60, 80]]
    """
    annotations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                row = [float(x) for x in line.split(',')]
                
                annotations.append(row)
        return annotations
    except FileNotFoundError:
        print("文件路径不存在")
        return []
    
def load_data_pred(file_path):
    """
    [x1, y1, x2, y2, confidence] 形式的预测结果文件
    返回嵌套列表，例如: [[0, 120, 30, 50], [1, 45, 60, 80]]
    """
    annotations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                row = [float(x) for x in line.split(',')]
                
                annotations.append(row)
        return annotations
    except FileNotFoundError:
        print("文件路径不存在")
        return []