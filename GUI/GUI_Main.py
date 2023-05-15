import ctypes
import sys

from QTGUI import *
from queue import Queue
# ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    GUI = PredictionGui()
    GUI.show()
    sys.exit(app.exec_())
    exit(0)
