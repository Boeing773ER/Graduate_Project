from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QAction, QStatusBar, QPlainTextEdit
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QTextDocument, QTextCursor, QTextCharFormat, QFont


class PredictionGui(QMainWindow):
    def __init__(self, parent=None):
        self.openfile = list()  # single file
        self.path_list = list()     # multiple file

        self.highlight = False  # high light or not
        window_icon = QIcon("./icon/text_editor.png")
        window_icon.addPixmap(QtGui.QPixmap("my.ico"), QIcon.Normal, QIcon.Off)

        # MainWindow
        super(PredictionGui, self).__init__(parent)
        self.setObjectName("MainWindow")
        self.setWindowTitle("Text Editor")
        self.setWindowIcon(window_icon)
        self.resize(800, 600)
        self.setMouseTracking(False)
        self.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.setToolTip("")

        # plainTextEdit
        self.plainTextEdit = QPlainTextEdit()
        self.editor_font = self.plainTextEdit.font()
        self.editor_font.setPointSize(20)
        self.plainTextEdit.setFont(self.editor_font)
        self.setCentralWidget(self.plainTextEdit)   # set it as central widget
        self.plainTextEdit.close()      # close, make it invisible
        self.plainTextEdit.textChanged.connect(lambda: self.text_changed())
        self.plainTextEdit.cursorPositionChanged.connect(lambda: self.cursor_pos_changed())
        #Cursor
        # self.text_cursor = self.plainTextEdit.textCursor()

        # menubar
        self.menubar = self.menuBar()
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        # menu
        menu_file = self.menubar.addMenu("File(&F)")
        menu_edit = self.menubar.addMenu("Edit(&E)")
        menu_coding = self.menubar.addMenu("Coding(&C)")
        menu_advanced = self.menubar.addMenu("Advanced(&A)")
        # QAction within menuBar
        m_save = QAction(QIcon("./icon/save.png"), "Save", self)
        m_save.setObjectName("M_Save")
        m_save.setShortcut("Ctrl+S")
        m_open = QAction(QIcon("./icon/open.png"), "Open", self)
        m_open.setObjectName("M_Open")
        m_open.setShortcut("Ctrl+O")
        m_new = QAction(QIcon("./icon/new file.png"), "New File", self)
        m_new.setObjectName("M_New")
        m_new.setShortcut("Ctrl+N")
        m_find = QAction(QIcon("./icon/find.png"), "Find", self)
        m_find.setObjectName("M_Find")
        m_find.setShortcut("Ctrl+F")
        m_replace = QAction(QIcon("./icon/replace.png"), "Replace", self)
        m_replace.setObjectName("M_Replace")
        m_replace.setShortcut("Ctrl+R")
        m_remove_highlight = QAction(QIcon("./icon/remove_tag.png"), "Remove Highlight", self)
        m_remove_highlight.setObjectName("M_Remove_Highlight")
        m_encode = QAction(QIcon("./icon/encode.png"), "Encode", self)
        m_encode.setObjectName("M_Encode")
        m_decode = QAction(QIcon("./icon/decode.png"), "Decode", self)
        m_decode.setObjectName("M_Decode")
        m_mul_search = QAction(QIcon("./icon/adv_search.png"), "Multi File Search", self)
        m_mul_search.setObjectName("M_Mul_Search")
        m_statistic = QAction(QIcon("./icon/statistic.png"), "Statistic", self)
        m_statistic.setObjectName("M_Statistic")
        # add action to menuBar
        menu_file.addAction(m_new)
        menu_file.addAction(m_open)
        menu_file.addSeparator()
        menu_file.addAction(m_save)
        menu_edit.addAction(m_find)
        menu_edit.addAction(m_replace)
        menu_edit.addSeparator()
        menu_edit.addAction(m_remove_highlight)
        menu_coding.addAction(m_encode)
        menu_coding.addAction(m_decode)
        menu_advanced.addAction(m_mul_search)
        menu_advanced.addAction(m_statistic)

        # toolBar
        self.toolBar = self.addToolBar("File")
        self.toolBar.setObjectName("toolBar")
        self.font_size = QComboBox()
        combo_font = self.font_size.font()
        combo_font.setPointSize(13)
        font_label = QLabel()
        font_label.setText("Font Size:")
        temp_font = font_label.font()
        temp_font.setBold(True)
        font_label.setFont(temp_font)
        self.font_size.setFont(combo_font)
        font_size_list = [10, 11, 12, 13, 15, 17, 20, 23, 26, 30]
        for i in range(10):
            self.font_size.addItem(str(font_size_list[i]))
            temp_font = combo_font
            temp_font.setPointSize(font_size_list[i])
            self.font_size.setItemData(i, temp_font, Qt.FontRole)
        self.font_size.setCurrentIndex(6)   # default value (3+1)*5 = 20
        self.font_size.setEditable(True)
        self.font_size.setMinimumWidth(100)
        self.font_size.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.font_size.currentTextChanged[str].connect(lambda: self.change_font_size())    # send content
        # add QAction to toolBar
        self.toolBar.addAction(m_new)
        self.toolBar.addAction(m_open)
        self.toolBar.addAction(m_save)
        self.toolBar.addSeparator()
        self.toolBar.addAction(m_find)
        self.toolBar.addAction(m_replace)
        self.toolBar.addAction(m_remove_highlight)
        self.toolBar.addSeparator()
        self.toolBar.addAction(m_encode)
        self.toolBar.addAction(m_decode)
        self.toolBar.addSeparator()
        self.toolBar.addAction(m_mul_search)
        self.toolBar.addAction(m_statistic)
        self.toolBar.addSeparator()
        self.toolBar.addWidget(font_label)
        self.toolBar.addWidget(self.font_size)

        # statusbar
        self.statusbar = QStatusBar()
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        # init component
        self.courser_pos = QLabel()
        self.courser_pos.setMinimumWidth(200)
        self.courser_pos.setAlignment(Qt.AlignCenter)
        self.courser_pos.setText("Bl:1 \t Col:1")
        self.sb_message = QLabel()
        self.sb_message.setMinimumWidth(200)
        self.sb_message.setAlignment(Qt.AlignCenter)
        self.word_count = QLabel()
        self.word_count.setMinimumWidth(100)
        self.word_count.setAlignment(Qt.AlignCenter)
        self.word_count.setText("0 Char")
        self.file_name = QLabel()
        self.file_name.setMinimumWidth(150)
        self.file_name.setAlignment(Qt.AlignCenter)
        self.file_name.setText("File:")
        self.statusbar.addPermanentWidget(self.sb_message)
        self.statusbar.addPermanentWidget(self.courser_pos)
        self.statusbar.addPermanentWidget(self.word_count)
        self.statusbar.addPermanentWidget(self.file_name)

        # connect
        m_new.triggered.connect(self.new_pressed)
        m_open.triggered.connect(self.open_pressed)
        m_save.triggered.connect(self.save_pressed)
        m_find.triggered.connect(self.find_pressed)
        m_replace.triggered.connect(self.replace_pressed)
        m_remove_highlight.triggered.connect(self.rem_hl_pressed)
        m_encode.triggered.connect(self.encode_pressed)
        m_decode.triggered.connect(self.decode_pressed)
        m_mul_search.triggered.connect(self.mul_search_pressed)
        m_statistic.triggered.connect(self.statistic_pressed)

        # testing
        # self.inverted_index()
