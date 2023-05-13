from copy import deepcopy, copy
from tokenize import String

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QAction, QStatusBar, QPlainTextEdit
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QTextDocument, QTextCursor, QTextCharFormat, QFont


class PredictionGui(QMainWindow):
    def __init__(self, parent=None):
        self.openfile = list()  # single file
        self.path_list = list()     # multiple file

        self.input_table_data = []  # data from input table

        self.day_length = 30

        self.highlight = False  # high light or not
        window_icon = QIcon("./icon/program.png")
        window_icon.addPixmap(QtGui.QPixmap("my.ico"), QIcon.Normal, QIcon.Off)

        # MainWindow
        super(PredictionGui, self).__init__(parent)
        self.setObjectName("MainWindow")
        self.setWindowTitle("COVID Prediction")
        self.setWindowIcon(window_icon)
        self.resize(800, 600)
        self.setMouseTracking(False)
        self.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.setToolTip("")

        # Init Page
        self.init_page = QFrame()

        self.normal_prompt = QFont()
        self.normal_prompt.setBold(True)
        self.normal_prompt.setPointSize(15)

        self.secondary_prompt = QFont()
        self.secondary_prompt.setPointSize(10)
        self.secondary_prompt.setItalic(True)

        # Layout
        # self.init_page_layout = QHBoxLayout()
        # self.file_upload_layout = self.init_page_layout.QVBoxLayout()
        # self.select_model_layout = self.init_page_layout.QVBoxLayout()

        # Select file sector
        # Label
        self.select_file_prompt = QLabel("Input statistic")
        self.select_file_prompt.setFont(self.normal_prompt)

        # self.select_file_explain = QLabel("File should a.csv file that contains 3 columns")
        self.select_file_explain = QLabel("Select a .csv file or input the data manually")
        self.select_file_explain.setFont(self.secondary_prompt)
        self.select_file_explain.setStyleSheet("color:grey")
        self.select_file_explain.setWordWrap(True)

        self.upload_file_prompt = QLabel("Data not loaded")

        # check box
        font = QFont()
        font.setPointSize(10)
        self.infected_checkbox = QCheckBox("Infected")
        self.infected_checkbox.setChecked(True)
        self.infected_checkbox.setFont(font)
        self.asymptom_checkbox = QCheckBox("Asymptom")
        self.asymptom_checkbox.setChecked(True)
        self.asymptom_checkbox.setFont(font)
        self.healed_checkbox = QCheckBox("Healed")
        self.healed_checkbox.setChecked(True)
        self.healed_checkbox.setFont(font)

        self.checkbox_layout = QVBoxLayout()
        self.checkbox_layout.addWidget(self.infected_checkbox)
        self.checkbox_layout.addWidget(self.asymptom_checkbox)
        self.checkbox_layout.addWidget(self.healed_checkbox)

        # spin box
        self.days_input = QSpinBox()
        self.days_input.setRange(1, 100)
        self.days_input.setSingleStep(1)
        self.days_input.setFont(font)
        self.days_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.days_input.setValue(self.day_length)
        self.days_input.valueChanged.connect(lambda: self.spinbox_value_change())

        # button
        button_font = QFont()
        button_font.setPointSize(12)
        button_font.setBold(True)
        self.select_file_button = QPushButton("Select File")
        self.select_file_button.setFont(button_font)
        self.select_file_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.select_file_button.clicked.connect(lambda: self.upload_file_pressed())

        self.manual_input_button = QPushButton("Manual Input")
        self.manual_input_button.setFont(button_font)
        self.manual_input_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.manual_input_button.clicked.connect(lambda: self.manual_input_pressed())

        self.upload_button_layout = QHBoxLayout()
        self.upload_button_layout.addWidget(self.select_file_button)
        self.upload_button_layout.addWidget(self.manual_input_button)

        # self.manual_input_prompt = QLabel("Manual statistic input")
        # self.manual_input_prompt.setFont(self.normal_prompt)

        # self.manual_input_explain = QLabel("You can also input the data manually")
        # self.manual_input_explain.setFont(self.secondary_prompt)
        # self.manual_input_explain.setStyleSheet("color:grey")
        # self.manual_input_prompt.setWordWrap(True)

        # layout
        self.file_upload_layout = QVBoxLayout()
        self.file_upload_layout.addWidget(self.select_file_prompt)
        self.file_upload_layout.addWidget(self.select_file_explain)
        self.file_upload_layout.addLayout(self.checkbox_layout)
        self.file_upload_layout.addWidget(self.days_input)
        self.file_upload_layout.addLayout(self.upload_button_layout)
        self.file_upload_layout.addWidget(self.upload_file_prompt)
        # self.file_upload_layout.addWidget(self.select_file_button)
        # self.file_upload_layout.addWidget(self.manual_input_prompt)
        # self.file_upload_layout.addWidget(self.manual_input_explain)
        # self.file_upload_layout.addWidget(self.manual_input_button)

        self.file_upload_layout.setContentsMargins(30, 80, 30, 150)
        # self.file_upload_layout.setSpacing(100)

        # Select model sector
        self.select_model_prompt = QLabel("Select model")
        self.select_model_prompt.setFont(self.normal_prompt)

        self.select_model_explain = QLabel("Select model explain here         ")
        self.select_model_explain.setFont(self.secondary_prompt)
        self.select_model_explain.setStyleSheet("color:grey")

        self.model_selector = QComboBox()
        self.model_selector.addItem("SIR")
        self.model_selector.addItem("SEIR")
        self.model_selector.addItem("SEIR-2")
        self.model_selector.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.model_sector_spacer = QSpacerItem(30, 70, QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.start_button = QPushButton("Start")
        self.start_button.setFont(self.normal_prompt)
        self.start_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.start_button.clicked.connect(lambda: self.start_pressed())
        # self.start_button.setStyleSheet("background:grey")

        # layout
        self.select_model_layout = QVBoxLayout()
        self.select_model_layout.addWidget(self.select_model_prompt)
        self.select_model_layout.addWidget(self.select_model_explain)
        self.select_model_layout.addWidget(self.model_selector)
        self.select_model_layout.addItem(self.model_sector_spacer)
        self.select_model_layout.addWidget(self.start_button)

        self.select_model_layout.setContentsMargins(30, 80, 30, 150)

        # =======================
        self.init_page_layout = QHBoxLayout()
        self.init_page_layout.addLayout(self.file_upload_layout, stretch=1)
        self.init_page_layout.addLayout(self.select_model_layout, stretch=1)
        # self.init_page_layout.setStretchFactor(self.file_upload_layout, 1)
        # self.init_page_layout.setStretchFactor(self.select_model_layout, 1)
        self.init_page.setLayout(self.init_page_layout)

        # Frame 2
        self.result_frame = QFrame()

        # button
        self.return_button = QPushButton("Back")
        self.return_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.return_button.clicked.connect(lambda: self.return2init())

        # ===============
        self.result_frame_layout = QHBoxLayout()
        self.result_frame_layout.addWidget(self.return_button)
        self.result_frame.setLayout(self.result_frame_layout)


        self.main_window_layout = QStackedLayout()
        self.central_widget = QFrame()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.main_window_layout)
        self.main_window_layout.addWidget(self.result_frame)
        self.main_window_layout.addWidget(self.init_page)
        self.main_window_layout.setCurrentIndex(1)


    def upload_file_pressed(self):
        print("in func upload_file_pressed")
        openfile_name = QFileDialog.getOpenFileName(self, caption='Upload File', filter="CSV Files(*.csv)")
        print(openfile_name)
        if openfile_name == ('', ''):
            print("open fail")
        else:
            self.openfile = openfile_name[0]
            self.upload_file_prompt.setText("File uploaded:"+self.openfile)

    def manual_input_pressed(self):
        print("in func manual input pressed")
        dialog = QDialog()
        dialog.setWindowTitle("Manual input")

        # table
        data_table = QTableWidget(3, self.day_length, dialog)
        header = ["Infected", "Asymptom", "Healed"]
        data_table.setVerticalHeaderLabels(header)
        # data_table.setHorizontalHeader().setSectionResizeMode(QHeaderView.Fixed)

        dialog_button = QDialogButtonBox()
        ok_button = QPushButton("OK")
        no_button = QPushButton("NO")
        dialog_button.addButton(ok_button, QDialogButtonBox.AcceptRole)
        dialog_button.addButton(no_button, QDialogButtonBox.RejectRole)
        dialog_button.accepted.connect(lambda: self.manual_input_accept(dialog, data_table))
        dialog_button.rejected.connect(lambda: self.manual_input_reject(dialog))

        # layout
        layout = QVBoxLayout()
        layout.addWidget(data_table)
        layout.addWidget(dialog_button)
        dialog.setLayout(layout)
        dialog.resize(800, 225)
        # dialog.exec()
        dialog.show()

    def start_pressed(self):
        print("in func start pressed")
        self.main_window_layout.setCurrentIndex(0)

    def stop_pressed(self):
        print("in func stop_pressed")

    def return2init(self):
        print("in func return2init")
        self.main_window_layout.setCurrentIndex(1)

    def save_result(self):
        print("in func save_result")

    def table_content_empty(self):
        print("in func table content empty")
        QMessageBox.warning(self, "Error", "Table Empty!", QMessageBox.Ok, QMessageBox.Ok)

    def table_content_illegal(self):
        print("in func table content illegal")
        QMessageBox.warning(self, "Error", "Table Content Illegal!", QMessageBox.Ok, QMessageBox.Ok)

    def manual_input_accept(self, dialog, table):
        print("in func manual input accept")
        row_count = table.model().rowCount()
        col_count = table.model().columnCount()
        for i in range(row_count):
            temp_list = []
            for j in range(col_count):
                if table.item(i, j) is None:
                    self.table_content_empty()
                    return
                else:
                    text = table.item(i, j).text()
                    try:
                        num = int(text)
                    except:
                        self.table_content_illegal()
                        return
            self.input_table_data.append(temp_list)
        print("out of for")
        self.upload_file_prompt.setText("Data inputted manually")
        dialog.destroy()

    def manual_input_reject(self, dialog):
        print("in func manual input reject")
        dialog.destroy()

    def spinbox_value_change(self):
        print("in func spinbox value change")
        temp = self.days_input.value()
        print(type(temp), temp)
        self.day_length = temp
