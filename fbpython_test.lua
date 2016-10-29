local py = require('fb.python')

py.exec([=[
import numpy as np
import re
import os
def foo(x):
    return x + 1

def load_img_path_list(path):
    """

    :param path: the test img folder
    :return:
    """
    p = re.compile(".*extract.jpg")
    list_path = os.listdir(path)
    # change to reg to match extension
    result = ["%s/%s" % (path, x) for x in list_path if p.match(x)]
    return result
print load_img_path_list("/home/chenqiang/data/human_post/test12")
  ]=])

print(py.eval('foo(a) + 10', {a = 42}))  -- prints 53
