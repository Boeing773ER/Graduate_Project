import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtChart import *

from GA import SEAIRVmodel, SEAIRmodel, SIARmodel
matplotlib.use('Qt5Agg')


class PredictionGui(QMainWindow):
    def __init__(self, parent=None):
        self.openfile = ""  # single file
        self.path_list = list()     # multiple file
        self.prediction_result = []
        self.prediction_params = []
        self.table_header = []
        self.input_table_data = []  # data from input table

        self.day_length = 30

        self.days_input = QSpinBox()
        self.data_table = QTableWidget(3, self.day_length)

        self.highlight = False  # high light or not
        window_icon = QIcon("./icon/program.png")
        window_icon.addPixmap(QtGui.QPixmap("my.ico"), QIcon.Normal, QIcon.Off)

        # MainWindow
        super(PredictionGui, self).__init__(parent)
        self.setObjectName("MainWindow")
        self.setWindowTitle("COVID Prediction")
        self.setWindowIcon(window_icon)
        self.resize(600, 600)
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

        # Select file sector
        # Label
        self.select_file_prompt = QLabel("Input statistic")
        self.select_file_prompt.setFont(self.normal_prompt)

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

        self.param_input_layout1 = QFormLayout()
        self.param_input_layout1.addRow("Data:", self.checkbox_layout)
        self.param_input_layout1.setSpacing(10)
        self.param_input_layout1.setLabelAlignment(Qt.AlignLeft)
        # temp_widget = self.param_input_layout1.labelForField(0)
        # temp_font = temp_widget.font()
        # temp_font.setPointSize(12)
        # temp_widget.setFont(temp_font)

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

        # spin box
        self.region_population_input = QSpinBox()
        self.region_population_input.setRange(10000, 100000000)
        self.region_population_input.setSingleStep(1000)
        self.region_population_input.setFont(font)
        self.region_population_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.region_population_input.setValue(10000)

        self.pre_days_input = QSpinBox()
        self.pre_days_input.setRange(1, 100)
        self.pre_days_input.setSingleStep(1)
        self.pre_days_input.setFont(font)
        self.pre_days_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.pre_days_input.setValue(10)
        # int_validator = QIntValidator(1, 1000000000)
        # self.pre_days_input.setValidator(int_validator)

        # model selector
        self.model_selector = QComboBox()
        self.model_selector.addItem("SIAR")
        self.model_selector.addItem("SEIAR")
        self.model_selector.addItem("SEIAR-V")
        self.model_selector.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # self.model_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        # self.model_selector.resize(100, 50)

        self.param_input_layout2 = QFormLayout()
        self.param_input_layout2.addRow("Pre Days:", self.pre_days_input)
        self.param_input_layout2.addRow("Population:", self.region_population_input)
        self.param_input_layout2.addRow("Model:", self.model_selector)
        # self.param_input_layout2.setSpacing(10)
        # self.param_input_layout2.setLabelAlignment(Qt.AlignLeft)

        # start button
        self.start_button = QPushButton("Start")
        self.start_button.setFont(self.normal_prompt)
        self.start_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.start_button.clicked.connect(lambda: self.start_pressed())

        # Lines
        self.Data_Frame = QFrame()
        self.Data_Frame.setFrameShape(QFrame.StyledPanel)
        self.Data_Frame_Layout = QVBoxLayout()
        self.Data_Frame_Layout.addLayout(self.param_input_layout1)
        self.Data_Frame_Layout.addLayout(self.upload_button_layout)
        self.Data_Frame.setLayout(self.Data_Frame_Layout)

        self.Param_Frame = QFrame()
        self.Param_Frame.setFrameShape(QFrame.StyledPanel)
        self.Param_Frame_Layout = QVBoxLayout()
        # self.Param_Frame_Layout.addLayout(self.param_input_layout2)
        # self.Param_Frame_Layout.addLayout(self.param_input_layout2)
        # self.Param_Frame.setLayout(self.Param_Frame_Layout)
        # self.Param_Frame.setLayout(self.param_input_layout2)

        # layout
        self.file_upload_layout = QVBoxLayout()
        self.file_upload_layout.addWidget(self.select_file_prompt)
        self.file_upload_layout.addWidget(self.select_file_explain)
        # self.file_upload_layout.addLayout(self.param_input_layout1)
        # self.file_upload_layout.addLayout(self.upload_button_layout)
        self.file_upload_layout.addWidget(self.Data_Frame)
        self.file_upload_layout.addLayout(self.param_input_layout2)
        # self.file_upload_layout.addWidget(self.Param_Frame)
        self.file_upload_layout.addWidget(self.start_button)
        self.file_upload_layout.addWidget(self.upload_file_prompt)

        self.file_upload_layout.setContentsMargins(30, 50, 30, 80)
        # self.file_upload_layout.setSpacing(100)

        """# Select model sector
        self.select_model_prompt = QLabel("Select model")
        self.select_model_prompt.setFont(self.normal_prompt)

        self.select_model_explain = QLabel("Select model")
        self.select_model_explain.setFont(self.secondary_prompt)
        self.select_model_explain.setStyleSheet("color:grey")

        self.model_sector_spacer = QSpacerItem(30, 70, QSizePolicy.Fixed, QSizePolicy.Fixed)

        # layout
        self.select_model_layout = QVBoxLayout()
        self.select_model_layout.addWidget(self.select_model_prompt)
        self.select_model_layout.addWidget(self.select_model_explain)
        self.select_model_layout.addWidget(self.model_selector)
        self.select_model_layout.addItem(self.model_sector_spacer)
        self.select_model_layout.addWidget(self.start_button)

        self.select_model_layout.setContentsMargins(30, 80, 30, 150)"""

        # =======================
        # self.init_page_layout = QHBoxLayout()
        # self.init_page_layout.addLayout(self.file_upload_layout, stretch=1)
        # self.init_page_layout.addLayout(self.select_model_layout, stretch=1)
        # self.init_page_layout.setStretchFactor(self.file_upload_layout, 1)
        # self.init_page_layout.setStretchFactor(self.select_model_layout, 1)
        self.init_page.setLayout(self.file_upload_layout)

        # Frame 2
        self.result_frame = QFrame()

        # button
        self.return_button = QPushButton("Back")
        self.return_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.return_button.clicked.connect(lambda: self.return2init())

        self.save_result_button = QPushButton("Save Result")
        self.save_result_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.save_result_button.clicked.connect(lambda :self.save_result())

        self.save_params_button = QPushButton("Save Params")
        self.save_params_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.save_params_button.clicked.connect(lambda: self.save_params())

        self.frame2_button_layout = QHBoxLayout()
        self.frame2_button_layout.addWidget(self.return_button)
        self.frame2_button_layout.addWidget(self.save_result_button)
        self.frame2_button_layout.addWidget(self.save_params_button)

        # 清屏
        plt.cla()
        # 获取绘图并绘制
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.canvas = FigureCanvas(self.fig)

        # ===============
        self.result_frame_layout = QVBoxLayout()
        self.result_frame_layout.addLayout(self.frame2_button_layout)
        self.result_frame_layout.addWidget(self.canvas)
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

    def select_days(self):
        print("in fun select days")

    def manual_input_pressed(self):
        print("in func manual input pressed")
        dialog = QDialog()
        dialog.setWindowTitle("Manual input")

        # table
        # self.data_table = QTableWidget(3, self.day_length, dialog)
        if self.infected_checkbox.isChecked():
            self.table_header.append("Infected")
        if self.asymptom_checkbox.isChecked():
            self.table_header.append("Asymptom")
        if self.healed_checkbox.isChecked():
            self.table_header.append("Healed")
        # header = ["Infected", "Asymptom", "Healed"]
        self.data_table.setRowCount(len(self.table_header))
        self.data_table.setVerticalHeaderLabels(self.table_header)
        # data_table.setHorizontalHeader().setSectionResizeMode(QHeaderView.Fixed)

        # spin box
        self.days_input.setRange(1, 100)
        self.days_input.setSingleStep(1)
        self.days_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.days_input.setValue(self.day_length)
        self.days_input.valueChanged.connect(lambda: self.spinbox_value_change())
        spinbox_layout = QFormLayout()
        spinbox_layout.addRow("Input Days:", self.days_input)

        dialog_button = QDialogButtonBox()
        ok_button = QPushButton("OK")
        no_button = QPushButton("NO")
        dialog_button.addButton(ok_button, QDialogButtonBox.AcceptRole)
        dialog_button.addButton(no_button, QDialogButtonBox.RejectRole)
        dialog_button.accepted.connect(lambda: self.manual_input_accept(dialog, self.data_table))
        dialog_button.rejected.connect(lambda: self.manual_input_reject(dialog))

        # layout
        layout = QVBoxLayout()
        layout.addWidget(self.data_table)
        layout.addLayout(spinbox_layout)
        layout.addWidget(dialog_button)
        dialog.setLayout(layout)
        # dialog.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.fixed)
        dialog.resize(800, 260)

        # dialog.exec()
        dialog.show()

    def start_pressed(self):
        print("in func start pressed")
        model_type = self.model_selector.currentText()
        infected_checked = self.infected_checkbox.isChecked()
        asymptom_checked = self.asymptom_checkbox.isChecked()
        healed_checked = self.healed_checkbox.isChecked()
        print("before get input")
        # region_population = int(self.region_population_input.text())
        # pre_days = int(self.pre_days_input.text())
        region_population = self.region_population_input.value()
        pre_days = self.pre_days_input.value()
        print("11111")
        data_type = []
        if self.infected_checkbox.isChecked():
            data_type.append(1)
        else:
            data_type.append(0)
        if self.asymptom_checkbox.isChecked():
            data_type.append(1)
        else:
            data_type.append(0)
        if self.healed_checkbox.isChecked():
            data_type.append(1)
        else:
            data_type.append(0)

        if model_type == "SEIAR-V":
            model = SEAIRVmodel(self.openfile, region_population, pre_days, data_type)
        elif model_type == "SEIAR":
            model = SEAIRmodel(self.openfile, region_population, pre_days, data_type)
        elif model_type == "SIAR":
            model = SIARmodel(self.openfile, region_population, pre_days, data_type)
        print("finish init")
        pre_t, self.prediction_result, self.prediction_params = model.start_GA(0)

        self.plot_curves(pre_t, self.prediction_result)
        self.main_window_layout.setCurrentIndex(0)

    def stop_pressed(self):
        print("in func stop_pressed")

    def return2init(self):
        print("in func return2init")
        self.main_window_layout.setCurrentIndex(1)

    def save_result(self):
        print("in func save_result")
        filename_save = QFileDialog.getSaveFileName(self, caption="Save New File", filter="*.csv")
        print(filename_save)
        if filename_save != ('', ''):
            print("Save_path:", filename_save)
            np.savetxt(filename_save[0], self.prediction_result, fmt="%d", delimiter=',')
        else:
            print("Save fail")

    def save_params(self):
        print("in func save_params")
        filename_save = QFileDialog.getSaveFileName(self, caption="Save New File", filter="*.csv")
        print(filename_save)
        if filename_save != ('', ''):
            print("Save_path:", filename_save)
            np.savetxt(filename_save[0], self.prediction_params, fmt="%d", delimiter=',')
        else:
            print("Save fail")

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
        temp_list = []
        for i in range(row_count):
            col_list = []
            for j in range(col_count):
                if table.item(i, j) is None:
                    self.table_content_empty()
                    return
                else:
                    text = table.item(i, j).text()
                    try:
                        num = int(text)
                        col_list.append(num)
                    except:
                        self.table_content_illegal()
                        return
            temp_list.append(col_list)
        table_input_array = np.array(temp_list).T
        dataframe = pd.DataFrame(table_input_array, columns=self.table_header)
        dataframe.to_csv("manual_input.csv", index=False, sep=',')
        self.openfile = "./manual_input.csv"
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
        self.data_table.setColumnCount(self.day_length)

    def plot_curves(self, pre_t, data):
        # 清屏
        plt.cla()
        # self.ax.set_xlim([-1, 6])
        # self.ax.set_ylim([-1, 6])
        plt.plot(pre_t, data[0, :], '--r', label='Pre_Inf')
        plt.plot(pre_t, data[1, :], '--y', label='Pre_Asy')
        plt.plot(pre_t, data[2, :], '--g', label='Pre_Rec')
        plt.legend()
        plt.grid()
        # self.ax.plot([0, 1, 2, 3, 4, 5], 'o--')
        # 将绘制好的图像设置为中心 Widget
        # self.setCentralWidget(cavans)
