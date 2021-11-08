# -*- coding: utf-8 -*-

__all__ = ['TemplateMainWindow']

from PySide2 import QtGui, QtWidgets


class TemplateMainWindow(QtWidgets.QMainWindow):
    """ This is a template QMainWindow subclass to be used with the qudi template_gui.py module.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Construct a simple GUI window with some widgets to tinker with
        # Initialize widgets
        self.reset_button = QtWidgets.QPushButton('Reset Counter')
        self.add_ten_button = QtWidgets.QPushButton('+10')
        self.sub_ten_button = QtWidgets.QPushButton('-10')
        self.count_spinbox = QtWidgets.QSpinBox()
        self.count_spinbox.setReadOnly(True)
        self.count_spinbox.setValue(0)
        self.label = QtWidgets.QLabel('Useless counter goes brrr...')
        font = QtGui.QFont()
        font.setBold(True)
        self.label.setFont(font)
        # arrange widgets in layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.sub_ten_button, 0, 0)
        layout.addWidget(self.count_spinbox, 0, 1)
        layout.addWidget(self.add_ten_button, 0, 2)
        layout.addWidget(self.reset_button, 1, 0, 1, 3)
        layout.setColumnStretch(1, 1)
        # Create dummy widget as main widget and set layout
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
