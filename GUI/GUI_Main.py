import ctypes
import sys

from QTGUI import *
from queue import Queue
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    text_editor = Text_Editor()
    text_editor.show()
    sys.exit(app.exec_())
    exit(0)

    """print(get_file_name("path/a/b/sgas.txt"))"""

    """show_str = "This it's gravity apple"
    word = "it"
    kmp_result = kmp_matching(show_str, word)
    for kmp_pos in kmp_result:
        if kmp_pos == 0:
            print(show_str[kmp_pos + len(word)], show_str[kmp_pos + len(word)].isalpha())
        elif kmp_pos == len(show_str)-1:
            print(show_str[kmp_pos - 1], show_str[kmp_pos - 1].isalpha())
        else:
            print(show_str[kmp_pos-1], show_str[kmp_pos-1].isalpha())
            print(show_str[kmp_pos + len(word)], show_str[kmp_pos + len(word)].isalpha())"""
    # and kmp_temp_result[j] + len(word) < len(show_str) \
    # and kmp_result[j] - 1 >= 0\

    """word_pos = []
    word_list = ['of', 'that', 'a']
    for word in word_list:
        temp = kmp_matching("Starts out with a opening scene that is a terrific example of absurd comedy", word)
        word_pos.extend(temp)
    print(word_pos)"""
    """str_a = "   ~ ~ aaa | bbb  &ccc"
    queue_a = nifix_to_postfix(str_a)
    while not queue_a.empty():
        print(queue_a.get())"""
